import pytest
import requests
import base64
from datetime import datetime

# ============== Health & Auth Tests ==============

class TestHealthAndAuth:
    """Health check and authentication endpoints"""

    def test_root_endpoint(self, base_url, api_client):
        """Test GET /api/ returns API info"""
        response = api_client.get(f"{base_url}/api/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'Django' in data['message']
        print("✓ Root endpoint working")

    def test_health_check(self, base_url, api_client):
        """Test GET /api/health"""
        response = api_client.get(f"{base_url}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['database'] == 'sqlite3'
        print("✓ Health check passed")

    def test_admin_login_success(self, base_url, api_client):
        """Test POST /api/auth/login with correct credentials"""
        response = api_client.post(f"{base_url}/api/auth/login", json={
            "email": "hello@gmail.com",
            "password": "hello"
        })
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        assert data['email'] == 'hello@gmail.com'
        assert data['role'] == 'admin'
        assert 'id' in data
        print("✓ Admin login successful")

    def test_login_invalid_credentials(self, base_url, api_client):
        """Test login with wrong password returns 401"""
        response = api_client.post(f"{base_url}/api/auth/login", json={
            "email": "hello@gmail.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        data = response.json()
        assert 'detail' in data
        print("✓ Invalid credentials rejected")

    def test_register_new_user(self, base_url, api_client):
        """Test POST /api/auth/register creates new user"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        response = api_client.post(f"{base_url}/api/auth/register", json={
            "name": "Test User",
            "email": f"test_{timestamp}@example.com",
            "password": "testpass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        assert data['email'] == f"test_{timestamp}@example.com"
        assert data['role'] == 'user'
        print("✓ User registration successful")

    def test_register_duplicate_email(self, base_url, api_client):
        """Test registering with existing email returns 400"""
        response = api_client.post(f"{base_url}/api/auth/register", json={
            "name": "Admin",
            "email": "hello@gmail.com",
            "password": "anypassword"
        })
        assert response.status_code == 400
        data = response.json()
        assert 'detail' in data
        print("✓ Duplicate email rejected")

    def test_register_short_password(self, base_url, api_client):
        """Test registration with short password returns 400"""
        response = api_client.post(f"{base_url}/api/auth/register", json={
            "name": "Test",
            "email": "test@example.com",
            "password": "123"
        })
        assert response.status_code == 400
        print("✓ Short password rejected")

    def test_auth_me_with_token(self, base_url, api_client, auth_headers):
        """Test GET /api/auth/me with valid token"""
        response = api_client.get(f"{base_url}/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data['email'] == 'hello@gmail.com'
        assert 'id' in data
        print("✓ Auth /me endpoint working")

    def test_auth_me_without_token(self, base_url, api_client):
        """Test GET /api/auth/me without token returns 401"""
        response = api_client.get(f"{base_url}/api/auth/me")
        assert response.status_code == 401
        print("✓ Unauthorized access blocked")

    def test_logout(self, base_url, api_client):
        """Test POST /api/auth/logout"""
        response = api_client.post(f"{base_url}/api/auth/logout")
        assert response.status_code == 200
        print("✓ Logout endpoint working")


# ============== Appointments CRUD Tests ==============

class TestAppointments:
    """Appointment CRUD operations"""

    def test_create_appointment_and_verify(self, base_url, api_client, auth_headers):
        """Test POST /api/appointments and verify with GET"""
        response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Smith",
            "specialty": "Cardiology",
            "location": "Main Hospital",
            "appointment_date": "2026-05-15",
            "appointment_time": "10:30",
            "notes": "Annual checkup"
        })
        assert response.status_code == 201
        data = response.json()
        assert data['doctor_name'] == 'TEST Dr. Smith'
        assert 'id' in data
        apt_id = data['id']

        # Verify with GET
        get_response = api_client.get(f"{base_url}/api/appointments/{apt_id}", headers=auth_headers)
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data['doctor_name'] == 'TEST Dr. Smith'
        print("✓ Appointment created and verified")

    def test_get_appointments_list(self, base_url, api_client, auth_headers):
        """Test GET /api/appointments returns list"""
        response = api_client.get(f"{base_url}/api/appointments", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Appointments list retrieved ({len(data)} items)")

    def test_update_appointment_and_verify(self, base_url, api_client, auth_headers):
        """Test PUT /api/appointments/{id} and verify changes"""
        # Create appointment
        create_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Jones",
            "appointment_date": "2026-06-01",
            "appointment_time": "14:00"
        })
        apt_id = create_response.json()['id']

        # Update
        update_response = api_client.put(f"{base_url}/api/appointments/{apt_id}", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Jones Updated",
            "is_completed": True
        })
        assert update_response.status_code == 200
        updated_data = update_response.json()
        assert updated_data['doctor_name'] == 'TEST Dr. Jones Updated'
        assert updated_data['is_completed'] == True

        # Verify with GET
        get_response = api_client.get(f"{base_url}/api/appointments/{apt_id}", headers=auth_headers)
        assert get_response.json()['is_completed'] == True
        print("✓ Appointment updated and verified")

    def test_delete_appointment_and_verify(self, base_url, api_client, auth_headers):
        """Test DELETE /api/appointments/{id} and verify 404"""
        # Create appointment
        create_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Delete",
            "appointment_date": "2026-07-01",
            "appointment_time": "09:00"
        })
        apt_id = create_response.json()['id']

        # Delete
        delete_response = api_client.delete(f"{base_url}/api/appointments/{apt_id}", headers=auth_headers)
        assert delete_response.status_code == 200

        # Verify 404
        get_response = api_client.get(f"{base_url}/api/appointments/{apt_id}", headers=auth_headers)
        assert get_response.status_code == 404
        print("✓ Appointment deleted and verified")

    def test_appointments_without_auth(self, base_url, api_client):
        """Test appointments endpoint without auth returns 401"""
        response = api_client.get(f"{base_url}/api/appointments")
        assert response.status_code == 401
        print("✓ Unauthorized access to appointments blocked")


# ============== Other CRUD Tests ==============

class TestSymptoms:
    """Symptom CRUD operations"""

    def test_create_symptom_and_verify(self, base_url, api_client, auth_headers):
        """Test POST /api/symptoms linked to appointment"""
        # Create appointment first
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Symptom",
            "appointment_date": "2026-05-20",
            "appointment_time": "11:00"
        })
        apt_id = apt_response.json()['id']

        # Create symptom
        symptom_response = api_client.post(f"{base_url}/api/symptoms", headers=auth_headers, json={
            "appointment_id": apt_id,
            "name": "TEST Headache",
            "severity": 7,
            "timing": "Morning",
            "duration": "2 hours",
            "is_new": True,
            "is_worsening": False
        })
        assert symptom_response.status_code == 201
        symptom_data = symptom_response.json()
        assert symptom_data['name'] == 'TEST Headache'
        assert symptom_data['severity'] == 7
        assert symptom_data['appointment_id'] == apt_id

        # Verify with GET
        get_response = api_client.get(f"{base_url}/api/symptoms", headers=auth_headers, params={'appointment_id': apt_id})
        assert get_response.status_code == 200
        symptoms = get_response.json()
        assert len(symptoms) > 0
        print("✓ Symptom created and verified")

    def test_update_symptom(self, base_url, api_client, auth_headers):
        """Test PUT /api/symptoms/{id}"""
        # Create appointment and symptom
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Update",
            "appointment_date": "2026-05-21",
            "appointment_time": "12:00"
        })
        apt_id = apt_response.json()['id']

        symptom_response = api_client.post(f"{base_url}/api/symptoms", headers=auth_headers, json={
            "appointment_id": apt_id,
            "name": "TEST Cough",
            "severity": 5
        })
        symptom_id = symptom_response.json()['id']

        # Update
        update_response = api_client.put(f"{base_url}/api/symptoms/{symptom_id}", headers=auth_headers, json={
            "severity": 8,
            "is_worsening": True
        })
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated['severity'] == 8
        assert updated['is_worsening'] == True
        print("✓ Symptom updated")


class TestFeelings:
    """Feeling CRUD operations"""

    def test_create_feeling_and_verify_by_date(self, base_url, api_client, auth_headers):
        """Test POST /api/feelings and GET by date"""
        today = datetime.now().strftime('%Y-%m-%d')
        response = api_client.post(f"{base_url}/api/feelings", headers=auth_headers, json={
            "health_score": 8,
            "energy_level": 7,
            "mood": "Good",
            "notes": "TEST Feeling great today",
            "date": today
        })
        assert response.status_code == 201
        data = response.json()
        assert data['health_score'] == 8
        assert data['energy_level'] == 7

        # Verify by date
        get_response = api_client.get(f"{base_url}/api/feelings/date/{today}", headers=auth_headers)
        assert get_response.status_code == 200
        print("✓ Feeling created and verified by date")


class TestQuestions:
    """Question CRUD operations"""

    def test_create_question_and_verify(self, base_url, api_client, auth_headers):
        """Test POST /api/questions with priority"""
        # Create appointment
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Question",
            "appointment_date": "2026-05-22",
            "appointment_time": "13:00"
        })
        apt_id = apt_response.json()['id']

        # Create question
        question_response = api_client.post(f"{base_url}/api/questions", headers=auth_headers, json={
            "appointment_id": apt_id,
            "text": "TEST What are the side effects?",
            "priority": 1,
            "is_answered": False
        })
        assert question_response.status_code == 201
        question_data = question_response.json()
        assert question_data['text'] == 'TEST What are the side effects?'
        assert question_data['priority'] == 1

        # Verify with GET
        get_response = api_client.get(f"{base_url}/api/questions", headers=auth_headers, params={'appointment_id': apt_id})
        assert get_response.status_code == 200
        print("✓ Question created and verified")


class TestNotes:
    """Note CRUD operations"""

    def test_create_note_and_verify(self, base_url, api_client, auth_headers):
        """Test POST /api/notes with category"""
        # Create appointment
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Note",
            "appointment_date": "2026-05-23",
            "appointment_time": "14:00"
        })
        apt_id = apt_response.json()['id']

        # Create note
        note_response = api_client.post(f"{base_url}/api/notes", headers=auth_headers, json={
            "appointment_id": apt_id,
            "title": "TEST Important Note",
            "content": "Remember to bring medical records",
            "category": "preparation"
        })
        assert note_response.status_code == 201
        note_data = note_response.json()
        assert note_data['title'] == 'TEST Important Note'
        assert note_data['category'] == 'preparation'

        # Verify with GET
        get_response = api_client.get(f"{base_url}/api/notes", headers=auth_headers, params={'appointment_id': apt_id})
        assert get_response.status_code == 200
        print("✓ Note created and verified")


class TestRecordings:
    """Recording CRUD operations"""

    def test_create_recording_and_verify(self, base_url, api_client, auth_headers):
        """Test POST /api/recordings and verify audio_base64 excluded from list"""
        # Create appointment
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Recording",
            "appointment_date": "2026-05-24",
            "appointment_time": "15:00"
        })
        apt_id = apt_response.json()['id']

        # Create recording
        recording_response = api_client.post(f"{base_url}/api/recordings", headers=auth_headers, json={
            "appointment_id": apt_id,
            "title": "TEST Voice Note",
            "duration_seconds": 120,
            "audio_base64": "base64encodedaudiodata"
        })
        assert recording_response.status_code == 201
        recording_data = recording_response.json()
        assert recording_data['title'] == 'TEST Voice Note'
        assert recording_data['duration_seconds'] == 120
        assert 'audio_base64' not in recording_data  # Should be excluded from list response
        recording_id = recording_data['id']

        # Verify with GET detail (should include audio_base64)
        get_response = api_client.get(f"{base_url}/api/recordings/{recording_id}", headers=auth_headers)
        assert get_response.status_code == 200
        detail_data = get_response.json()
        assert 'audio_base64' in detail_data
        print("✓ Recording created and verified")


class TestSummary:
    """Summary endpoint test"""

    def test_appointment_summary(self, base_url, api_client, auth_headers):
        """Test GET /api/appointments/{id}/summary aggregates all data"""
        # Create appointment
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Summary",
            "appointment_date": "2026-05-25",
            "appointment_time": "16:00"
        })
        apt_id = apt_response.json()['id']

        # Create related data
        api_client.post(f"{base_url}/api/symptoms", headers=auth_headers, json={
            "appointment_id": apt_id,
            "name": "TEST Summary Symptom",
            "severity": 6
        })
        api_client.post(f"{base_url}/api/questions", headers=auth_headers, json={
            "appointment_id": apt_id,
            "text": "TEST Summary Question",
            "priority": 1
        })

        # Get summary
        summary_response = api_client.get(f"{base_url}/api/appointments/{apt_id}/summary", headers=auth_headers)
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert 'appointment' in summary
        assert 'symptoms' in summary
        assert 'questions' in summary
        assert 'notes' in summary
        assert 'feelings' in summary
        assert 'recordings' in summary
        assert 'personalization_profile' in summary
        assert len(summary['symptoms']) > 0
        assert len(summary['questions']) > 0
        assert 'llm_input_coverage' in summary
        print("✓ Summary endpoint working")

    def test_generate_one_pager(self, base_url, api_client, auth_headers):
        """Test POST /api/appointments/{id}/generate-one-pager returns one-pager + coverage."""
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. OnePager",
            "appointment_date": "2026-05-26",
            "appointment_time": "11:00"
        })
        apt_id = apt_response.json()['id']
        api_client.post(f"{base_url}/api/symptoms", headers=auth_headers, json={
            "appointment_id": apt_id,
            "name": "TEST OnePager Symptom",
            "severity": 5
        })

        response = api_client.post(
            f"{base_url}/api/appointments/{apt_id}/generate-one-pager",
            headers=auth_headers,
            json={"view_mode": "standard", "force": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert 'one_pager' in data
        assert data['one_pager'].get('headline')
        assert 'source' in data
        assert 'llm_input_coverage' in data
        assert 'llm_context_warnings' in data
        print("✓ Generate one-pager endpoint working")

    def test_export_summary_pdf(self, base_url, api_client, auth_headers):
        """Test POST /api/appointments/{id}/export-pdf returns base64-encoded PDF"""
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Export",
            "appointment_date": "2026-06-01",
            "appointment_time": "10:00"
        })
        apt_id = apt_response.json()['id']

        response = api_client.post(
            f"{base_url}/api/appointments/{apt_id}/export-pdf",
            headers=auth_headers,
            json={
                "layout_style": "compact",
                "font_size": "normal",
                "include_personalization": True,
                "date_format": "long",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert 'filename' in data
        assert data['filename'].endswith('.pdf')
        assert 'content_base64' in data
        assert 'download_url' in data
        assert data['personalization_applied'] is True
        assert data['ai_personalization_requested'] is True
        assert isinstance(data['ai_personalization_used'], bool)

        decoded = base64.b64decode(data['content_base64'])
        assert decoded.startswith(b'%PDF')

        # PART 5 — verify the download_url path independently returns a
        # real PDF (application/pdf, %PDF header) with auth required, and
        # that content_base64 (relied on by other tests/consumers) is untouched.
        download_url = data['download_url']
        dl_response = api_client.get(download_url, headers=auth_headers)
        assert dl_response.status_code == 200
        assert dl_response.headers.get('content-type', '').startswith('application/pdf')
        assert dl_response.content.startswith(b'%PDF')
        print("✓ PDF export endpoint working")

    def test_export_summary_pdf_download_requires_auth(self, base_url, api_client, auth_headers):
        """download_url must not be publicly fetchable without a valid bearer token."""
        apt_response = api_client.post(f"{base_url}/api/appointments", headers=auth_headers, json={
            "doctor_name": "TEST Dr. Export Auth",
            "appointment_date": "2026-06-02",
            "appointment_time": "09:00",
        })
        apt_id = apt_response.json()['id']

        export_response = api_client.post(
            f"{base_url}/api/appointments/{apt_id}/export-pdf",
            headers=auth_headers,
            json={"use_ai_personalization": False},
        )
        assert export_response.status_code == 200
        download_url = export_response.json()['download_url']

        unauthenticated = api_client.get(download_url)
        assert unauthenticated.status_code in (401, 403, 404)
        print("✓ PDF download_url requires authentication")
