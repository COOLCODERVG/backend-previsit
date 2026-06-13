from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0004_personalization_family_history"),
    ]

    operations = [
        migrations.AddField(
            model_name="appointment",
            name="visit_summary_cache",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
