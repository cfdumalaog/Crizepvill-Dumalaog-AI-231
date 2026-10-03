#!/usr/bin/env bash
set -euo pipefail

archive="${1:?usage: install_me2_candidate_on_pi.sh ARCHIVE EXPECTED_SHA256}"
expected_sha="${2:?usage: install_me2_candidate_on_pi.sh ARCHIVE EXPECTED_SHA256}"
target="/home/dalmacio/Desktop/me2-dataset-candidate"
venv="/home/dalmacio/.venvs/tinyvcm-rpi5"

if [[ ! -f "$archive" ]]; then
  echo "Candidate archive does not exist: $archive" >&2
  exit 2
fi
actual_sha="$(sha256sum -- "$archive" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Candidate archive SHA-256 mismatch" >&2
  exit 3
fi
if [[ -e "$target" ]]; then
  echo "Refusing to replace existing candidate folder: $target" >&2
  exit 4
fi
if [[ ! -x "$venv/bin/python" ]]; then
  echo "Expected shared Pi runtime is missing: $venv/bin/python" >&2
  exit 5
fi

stage="$(mktemp -d /tmp/me2-candidate-install.XXXXXX)"
case "$stage" in
  /tmp/me2-candidate-install.*) ;;
  *) echo "Unexpected staging directory: $stage" >&2; exit 6 ;;
esac
cleanup() {
  case "$stage" in
    /tmp/me2-candidate-install.*) rm -rf -- "$stage" ;;
    *) echo "Refusing to remove unexpected staging path: $stage" >&2 ;;
  esac
}
trap cleanup EXIT

unzip -q -- "$archive" -d "$stage"
bundle="$stage/me2-dataset-candidate"
if [[ ! -f "$bundle/bundle_manifest.json" || ! -f "$bundle/launch-vcm.sh" ]]; then
  echo "Archive does not contain the expected candidate bundle" >&2
  exit 7
fi
python3 - "$bundle/deployment.json" <<'PY'
import json, sys
settings = json.load(open(sys.argv[1], encoding="utf-8"))
assert settings["port"] == 7865, settings
assert settings["gpio"] is False, settings
assert settings["autostart"] is False, settings
assert settings["intent_confidence_threshold"] == 0.76, settings
assert settings["intent_margin_threshold"] == 0.0, settings
PY
mv -- "$bundle" "$target"
chmod 755 "$target/launch-vcm.sh"
cd "$target"
"$venv/bin/python" verify_personalized_rpi.py > install_verification.json
printf 'Installed %s\nArchive SHA-256: %s\n' "$target" "$actual_sha"
cat install_verification.json
