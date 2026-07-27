from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0011_remove_recording_audio_storage'),
    ]

    operations = [
        migrations.AddField(
            model_name='recording',
            name='visit_summary',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='recording',
            name='visit_summary_status',
            field=models.CharField(
                choices=[
                    ('pending', 'pending'),
                    ('processing', 'processing'),
                    ('completed', 'completed'),
                    ('failed', 'failed'),
                ],
                default='pending',
                help_text='pending|processing|completed|failed',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='recording',
            name='visit_summary_error',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
    ]
