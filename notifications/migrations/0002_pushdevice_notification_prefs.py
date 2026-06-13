from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("notifications", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="pushdevice",
            name="notification_prefs",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
