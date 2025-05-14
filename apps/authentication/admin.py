# apps/authentication/admin.py
from django.contrib import admin
from .models import ViewHistory

@admin.register(ViewHistory)
class ViewHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'content_type', 'result', 'confidence', 'created_at')
    list_filter = ('content_type', 'result', 'created_at')
    search_fields = ('user__username', 'content')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')