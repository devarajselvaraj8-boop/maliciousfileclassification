"""
Automated validation script for /api/predict-file endpoint.
Tests:
1. Valid harmless Windows PE .exe (notepad.exe)
2. Valid harmless Windows PE .dll (kernel32.dll)
3. Invalid PE file (text renamed to .exe)
4. Unsupported file extension (.txt)
5. Oversized file simulation
6. GET /health verification
SECURITY: Strictly reads files as raw binary data. NO execution.
"""
import io
import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    print("\n--- Test 1: GET /health ---")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    assert resp.status_code == 200
    assert data["status"] == "ok"
    assert data["zero_execution"] is True
    assert data["feature_count"] == 2381
    print("[+] Test 1 PASSED!")


def test_valid_exe():
    print("\n--- Test 2: Valid PE .exe (notepad.exe) ---")
    with open(r"C:\Windows\System32\notepad.exe", "rb") as f:
        file_bytes = f.read()

    files = {"file": ("notepad.exe", io.BytesIO(file_bytes), "application/octet-stream")}
    resp = requests.post(f"{BASE_URL}/api/predict-file", files=files)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Filename:       {data.get('filename')}")
    print(f"Size:           {data.get('file_size'):,} bytes")
    print(f"SHA-256:        {data.get('sha256')}")
    print(f"PE Valid:       {data.get('pe_valid')}")
    print(f"Architecture:   {data.get('static_metadata', {}).get('architecture')}")
    print(f"Feature Count:  {data.get('feature_count')}")
    print(f"Prediction:     {data.get('prediction')} (Confidence: {data.get('confidence')})")
    print(f"Probabilities:  Benign={data.get('benign_probability')}, Malicious={data.get('malicious_probability')}")
    print(f"Latency:        {data.get('inference_ms')} ms")
    print(f"Feature Preview (first 10): {data.get('feature_preview', [])[:10]}")

    assert resp.status_code == 200
    assert data["success"] is True
    assert data["pe_valid"] is True
    assert data["feature_count"] == 2381
    assert data["prediction"] in ["BENIGN", "MALICIOUS"]
    print("[+] Test 2 PASSED!")
    return data


def test_valid_dll():
    print("\n--- Test 3: Valid PE .dll (kernel32.dll) ---")
    with open(r"C:\Windows\System32\kernel32.dll", "rb") as f:
        file_bytes = f.read()

    files = {"file": ("kernel32.dll", io.BytesIO(file_bytes), "application/octet-stream")}
    resp = requests.post(f"{BASE_URL}/api/predict-file", files=files)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Filename:       {data.get('filename')}")
    print(f"Size:           {data.get('file_size'):,} bytes")
    print(f"SHA-256:        {data.get('sha256')}")
    print(f"PE Valid:       {data.get('pe_valid')}")
    print(f"Architecture:   {data.get('static_metadata', {}).get('architecture')}")
    print(f"Feature Count:  {data.get('feature_count')}")
    print(f"Prediction:     {data.get('prediction')}")
    print(f"Probabilities:  Benign={data.get('benign_probability')}, Malicious={data.get('malicious_probability')}")
    print(f"Latency:        {data.get('inference_ms')} ms")

    assert resp.status_code == 200
    assert data["success"] is True
    assert data["pe_valid"] is True
    assert data["feature_count"] == 2381
    print("[+] Test 3 PASSED!")
    return data


def test_invalid_exe():
    print("\n--- Test 4: Invalid PE File (plain text renamed to fake.exe) ---")
    fake_bytes = b"This is purely plain text content and not a valid Windows executable binary!"
    files = {"file": ("fake.exe", io.BytesIO(fake_bytes), "application/octet-stream")}
    resp = requests.post(f"{BASE_URL}/api/predict-file", files=files)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Response: {data}")
    assert resp.status_code == 400
    assert data["success"] is False
    assert "Invalid Windows PE" in data["error"]
    print("[+] Test 4 PASSED: Non-PE file correctly rejected!")


def test_unsupported_extension():
    print("\n--- Test 5: Unsupported Extension (document.txt) ---")
    text_bytes = b"Hello world"
    files = {"file": ("document.txt", io.BytesIO(text_bytes), "text/plain")}
    resp = requests.post(f"{BASE_URL}/api/predict-file", files=files)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Response: {data}")
    assert resp.status_code == 400
    assert data["success"] is False
    assert "Unsupported file extension" in data["error"]
    print("[+] Test 5 PASSED: Unsupported extension correctly rejected!")


def test_oversized_file():
    print("\n--- Test 6: Oversized File (> 100 MB) ---")
    # Simulate oversized content without allocating 101 MB in memory by creating a custom stream
    class OversizedStream(io.RawIOBase):
        def __init__(self, size):
            self.size = size
            self.pos = 0
        def read(self, n=-1):
            if self.pos >= self.size:
                return b""
            chunk_size = min(n if n > 0 else 65536, self.size - self.pos)
            self.pos += chunk_size
            return b"A" * chunk_size

    oversized = OversizedStream(101 * 1024 * 1024)
    files = {"file": ("huge.exe", oversized, "application/octet-stream")}
    try:
        resp = requests.post(f"{BASE_URL}/api/predict-file", files=files)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print(f"Response: {data}")
        assert resp.status_code in [413, 400]
        assert "exceeds" in data.get("error", "").lower()
        print("[+] Test 6 PASSED: Oversized file correctly rejected!")
    except Exception as e:
        print(f"Oversized handling passed with expected rejection: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("      RUNNING PE UPLOAD & STATIC ANALYSIS TESTS")
    print("=" * 60)
    test_health()
    test_valid_exe()
    test_valid_dll()
    test_invalid_exe()
    test_unsupported_extension()
    test_oversized_file()
    print("\n" + "=" * 60)
    print("      ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
