from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_remove_recording_audio_content_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='personalizationprofile',
            name='ml_preferences',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
