#!/usr/bin/env bash
# Deploy the recommender API to GCP Cloud Run.
#
# Prerequisites (see docs/deployment.md):
#   - gcloud CLI installed and authenticated:  gcloud auth login
#   - a project with billing enabled:          gcloud config set project <ID>
#   - APIs enabled: run, artifactregistry, cloudbuild
#
# Usage:
#   ./scripts/deploy_cloudrun.sh                       # uses defaults below
#   REGION=europe-west1 ./scripts/deploy_cloudrun.sh
set -euo pipefail

# The Cloud SDK ships its own Python. On Windows/Git Bash, gcloud otherwise
# picks up whatever python is first on PATH — here a Windows Store Python 3.9,
# which current gcloud refuses to run on ("no longer supported by gcloud").
# Point at the bundled interpreter when one exists and CLOUDSDK_PYTHON is unset.
if [[ -z "${CLOUDSDK_PYTHON:-}" ]]; then
  for candidate in     "${LOCALAPPDATA:-$HOME/AppData/Local}/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe"     "/c/Users/${USER:-$USERNAME}/AppData/Local/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe"
  do
    if [[ -x "${candidate}" ]]; then
      export CLOUDSDK_PYTHON="$(cygpath -w "${candidate}" 2>/dev/null || echo "${candidate}")"
      break
    fi
  done
fi

command -v gcloud >/dev/null || {
  echo "gcloud not on PATH. Add it, e.g.:" >&2
  echo "  export PATH=\"\$PATH:/c/Users/\$USERNAME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin\"" >&2
  exit 1
}

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-short-video-recsys}"
REPO="${REPO:-recsys}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:latest"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "No GCP project set. Run: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi

echo "Project ${PROJECT} | region ${REGION} | service ${SERVICE}"

# One-time setup; both commands are idempotent.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  --project "${PROJECT}"

gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Short-video recsys images" \
  --project "${PROJECT}" 2>/dev/null || echo "Artifact Registry repo already exists"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# The deploy stage bakes the ~10 MB of trained artifacts into the image, so the
# service has no runtime dependency on object storage or credentials.
echo "Building deploy image ..."
docker build -f Dockerfile.cpu --target deploy -t "${IMAGE}" .
docker push "${IMAGE}"

# Sizing notes:
#   --memory 2Gi   torch plus the loaded artifacts; 1Gi is tight once FAISS and
#                  the embeddings are resident.
#   --cpu 1        the ranker forward pass is single-threaded at this batch size.
#   --min-instances 0  scale to zero. Measured cold start is ~2 s locally; on
#                  Cloud Run add image pull on the first start of a revision.
#                  Set to 1 if a demo link must never be slow — note that idle
#                  instances bill outside the free tier.
#   --concurrency 40   FastAPI is async but scoring is CPU-bound; this caps
#                  queueing behind the ranker rather than letting latency climb.
echo "Deploying ..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 4 \
  --concurrency 40 \
  --timeout 60 \
  --project "${PROJECT}"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --project "${PROJECT}" --format 'value(status.url)')"

echo
echo "Deployed: ${URL}"
echo
echo "Smoke test:"
echo "  curl ${URL}/health"
echo "  curl -X POST ${URL}/recommend -H 'Content-Type: application/json' -d '{\"user_id\": 0, \"top_k\": 5}'"
echo
echo "Measure latency (p95 over a real distribution, not one request):"
echo "  python scripts/loadtest.py --url ${URL} --requests 500 --out logs/loadtest_cloudrun.json"
