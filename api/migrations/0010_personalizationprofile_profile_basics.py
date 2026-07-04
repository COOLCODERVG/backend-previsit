from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0009_personalizationprofile_average_appointment_minutes'),
    ]

    operations = [
        migrations.AddField(
            model_name='personalizationprofile',
            name='age',
            field=models.CharField(blank=True, default='', max_length=10),
        ),
        migrations.AddField(
            model_name='personalizationprofile',
            name='gender',
            field=models.CharField(blank=True, default='', max_length=30),
        ),
        migrations.AddField(
            model_name='personalizationprofile',
            name='region',
            field=models.CharField(blank=True, default='', max_length=64),
        ),
    ]
