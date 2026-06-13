from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0003_recording_object_storage"),
    ]

    operations = [
        migrations.AddField(
            model_name="personalizationprofile",
            name="family_history",
            field=models.TextField(blank=True, default=""),
        ),
    ]
