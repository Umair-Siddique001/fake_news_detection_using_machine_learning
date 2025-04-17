from django.urls import path
from .views import LandingView



urlpatterns = [
    path(
        "",
      LandingView.as_view(template_name="landing.html"),
        name="landing",
    )
]
