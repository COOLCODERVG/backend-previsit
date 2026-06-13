from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PersonalizationProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('main_reason', models.TextField(blank=True, default='')),
                ('condition_stage', models.CharField(blank=True, choices=[('new', 'New'), ('ongoing', 'Ongoing'), ('not_sure', 'Not sure')], default='', max_length=20)),
                ('biggest_concern', models.TextField(blank=True, default='')),
                ('prepared_items', models.JSONField(blank=True, default=list)),
                ('appointment_outcome', models.CharField(blank=True, choices=[('clear_diagnosis', 'Clear diagnosis'), ('next_steps_plan', 'Next steps / treatment plan'), ('tests_or_referrals', 'Tests or referrals'), ('heard_understood', 'Just to be heard / understood')], default='', max_length=30)),
                ('is_completed', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='personalization', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'personalization_profiles',
            },
        ),
    ]
