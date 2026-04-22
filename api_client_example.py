"""
Minimal end-to-end client: upload -> poll -> download.

Usage:
    python api_client_example.py http://<pod-ip>:8000 path/to/face.png path/to/speech.wav
"""
import sys
import time
import requests

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    base, img_path, aud_path = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]

    print(f"[client] POST {base}/jobs")
    with open(img_path, "rb") as fi, open(aud_path, "rb") as fa:
        r = requests.post(
            f"{base}/jobs",
            files={"image": fi, "audio": fa},
            data={"prompt": "A person talking to the camera.",
                  "size": "infinitetalk-480",
                  "mode": "clip"},
            timeout=120,
        )
    r.raise_for_status()
    job = r.json()
    jid = job["job_id"]
    print(f"[client] created job {jid}")

    last_stage = None
    while True:
        s = requests.get(f"{base}/jobs/{jid}", timeout=30).json()
        if s["stage"] != last_stage:
            print(f"[client] stage={s['stage']} progress={s['progress']:.2f} status={s['status']}")
            last_stage = s["stage"]
        if s["status"] in ("completed", "failed"):
            break
        time.sleep(5)

    if s["status"] == "failed":
        print(f"[client] FAILED: {s.get('error')}")
        sys.exit(2)

    out = f"output_{jid}.mp4"
    print(f"[client] downloading -> {out}")
    with requests.get(f"{base}/jobs/{jid}/download", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    print(f"[client] saved {out}")

if __name__ == "__main__":
    main()
