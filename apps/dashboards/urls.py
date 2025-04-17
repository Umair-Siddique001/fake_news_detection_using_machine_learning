from django.urls import path
from .views import DashboardsView
from . import views



urlpatterns = [
    path(
        "dashboard/",
        DashboardsView.as_view(template_name="dashboard_analytics.html"),
        name="index",
    ),
    path("analyze-pdf/", views.analyze_pdf, name="analyze-pdf"),
    path("analyze-text/", views.analyze_text, name="analyze-text"),
    path("analyze-url/", views.analyze_url, name="analyze-url"),
    path("analyze-file/", views.analyze_pdf, name="analyze-file"),
]
