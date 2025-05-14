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
    path('view-history/', views.view_history, name='view-history'),
    path('clear-history/', views.clear_history, name='clear-history'),
    path('delete-history/<int:entry_id>/', views.delete_history, name='delete-history'),
]
