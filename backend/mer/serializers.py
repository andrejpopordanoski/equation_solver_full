from rest_framework import serializers
from .models import Latex


class LatexSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    latex_string = serializers.CharField(required=False, allow_blank=True, max_length=100)

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return Latex.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.latex_string = validated_data.get('latex_string', instance.latex_string)
        instance.save()
        return instance


