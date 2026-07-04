from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0008_recording_audio_content_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='personalizationprofile',
            name='average_appointment_minutes',
            field=models.PositiveIntegerField(default=30),
        ),
    ]
