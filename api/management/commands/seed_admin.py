import os
from django.core.management.base import BaseCommand
from api.models import User

class Command(BaseCommand):
    help = 'Seed admin user from environment variables'

    def handle(self, *args, **kwargs):
        email = os.environ.get('ADMIN_EMAIL', 'hello@gmail.com')
        password = os.environ.get('ADMIN_PASSWORD', 'hello')
        if not User.objects.filter(email=email).exists():
            User.objects.create_user(email=email, password=password, name='Admin', role='admin')
            self.stdout.write(f'Admin created: {email}')
        else:
            user = User.objects.get(email=email)
            if not user.check_password(password):
                user.set_password(password)
                user.save()
                self.stdout.write(f'Admin password updated: {email}')
            else:
                self.stdout.write(f'Admin exists: {email}')
