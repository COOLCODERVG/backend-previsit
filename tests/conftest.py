import pytest
import requests
import os

@pytest.fixture(scope="session")
def base_url():
    url = os.environ.get('EXPO_PUBLIC_BACKEND_URL')
    if not url:
        pytest.fail('EXPO_PUBLIC_BACKEND_URL not set in environment')
    return url.rstrip('/')

@pytest.fixture(scope="session")
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session

@pytest.fixture(scope="session")
def admin_token(base_url, api_client):
    """Login as admin and return access token"""
    response = api_client.post(f"{base_url}/api/auth/login", json={
        "email": "hello@gmail.com",
        "password": "hello"
    })
    if response.status_code != 200:
        pytest.fail(f'Admin login failed: {response.status_code} {response.text}')
    data = response.json()
    return data.get('access_token')

@pytest.fixture
def auth_headers(admin_token):
    """Return headers with Bearer token"""
    return {"Authorization": f"Bearer {admin_token}"}
