"""Download pinned Three.js runtime once; no account or paid APIs involved."""
from pathlib import Path
import hashlib
import json
import requests

VERSION = "0.180.0"
BASE = f"https://cdn.jsdelivr.net/npm/three@{VERSION}/"
FILES = {
    "build/three.module.js": "three.module.js",
    "build/three.core.js": "three.core.js",
    "examples/jsm/controls/OrbitControls.js": "addons/controls/OrbitControls.js",
    "examples/jsm/controls/TransformControls.js": "addons/controls/TransformControls.js",
    "examples/jsm/loaders/GLTFLoader.js": "addons/loaders/GLTFLoader.js",
    "examples/jsm/loaders/DRACOLoader.js": "addons/loaders/DRACOLoader.js",
    "examples/jsm/utils/BufferGeometryUtils.js": "addons/utils/BufferGeometryUtils.js",
    "examples/jsm/environments/RoomEnvironment.js": "addons/environments/RoomEnvironment.js",
    "examples/jsm/libs/meshopt_decoder.module.js": "addons/libs/meshopt_decoder.module.js",
    "examples/jsm/libs/draco/gltf/draco_decoder.js": "draco/draco_decoder.js",
    "examples/jsm/libs/draco/gltf/draco_decoder.wasm": "draco/draco_decoder.wasm",
    "examples/jsm/libs/draco/gltf/draco_wasm_wrapper.js": "draco/draco_wasm_wrapper.js",
    "LICENSE": "LICENSE-THREE.txt",
}


def main():
    target = Path(__file__).resolve().parents[1] / "unicompose3d/static/vendor"
    hashes = {}
    for source, relative in FILES.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = requests.get(BASE + source, timeout=90)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"Empty file: {relative}")
            path.write_bytes(response.content)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(relative)
    (target / "manifest.json").write_text(json.dumps({"three_version": VERSION, "sha256": hashes}, indent=2), encoding="utf-8")
    print("Three.js ready; browser runtime does not require a CDN.")


if __name__ == "__main__":
    main()
