from __future__ import annotations

from rest_framework import serializers


class VectorSearchSerializer(serializers.Serializer):
    """Provide `query_vector` and/or `query_text` (text is encoded when embedding service is configured)."""

    query_vector = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        allow_empty=False,
    )
    query_text = serializers.CharField(required=False, allow_blank=True)
    top_k = serializers.IntegerField(default=5, min_value=1, max_value=20)

    def validate(self, attrs):
        qv = attrs.get("query_vector")
        qt = (attrs.get("query_text") or "").strip()
        if not qv and not qt:
            raise serializers.ValidationError("Provide query_vector and/or query_text")
        return attrs


class PreferenceContextUpsertSerializer(serializers.Serializer):
    source = serializers.CharField(required=False, allow_blank=True)
    content = serializers.CharField()
    semantic_vec = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        allow_empty=False,
    )
    representation_meta = serializers.JSONField(required=False, default=dict)


class SteeringVectorUpsertSerializer(serializers.Serializer):
    context_id = serializers.UUIDField()
    layer = serializers.IntegerField(default=0, min_value=0, max_value=200)
    steering_vec = serializers.ListField(child=serializers.FloatField(), allow_empty=False)
    norm = serializers.FloatField(required=False, allow_null=True)
