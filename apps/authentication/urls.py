from django.urls import path
from . import views


urlpatterns = [
    path('login/', views.login_user, name='auth-login-basic'),
    path('register/', views.register_user, name='auth-register-basic'),
    path('forgot-password/', views.forgot_password, name='auth-forgot-password-basic'),
    path('logout/', views.logout_user, name='auth-logout'),
    path('profile/', views.profile_view, name='auth-profile'),
    path('profile/update/', views.update_profile, name='update-profile'),
    path('change-password/', views.change_password, name='auth-change-password'),
]
