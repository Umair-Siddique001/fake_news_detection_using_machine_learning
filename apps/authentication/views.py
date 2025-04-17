from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.models import User
from django.views.generic import TemplateView
from web_project import TemplateLayout
from django.contrib import messages
from web_project.template_helpers.theme import TemplateHelper
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes


"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to auth/urls.py file for more pages.
"""


class AuthView(TemplateView):
    # Predefined function
    def get_context_data(self, **kwargs):
        # A function to init the global layout. It is defined in web_project/__init__.py file
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        # Update the context
        context.update(
            {
                "layout_path": TemplateHelper.set_layout("layout_blank.html", context),
            }
        )

        return context

def register_user(request):
    context = {
        "layout_path": TemplateHelper.set_layout("layout_blank.html", {}),
    }
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        terms = request.POST.get('terms')
        
        # Store form data in context to repopulate the form
        context.update({
            'form_data': {
                'username': username,
                'email': email,
                'terms': terms
            }
        })
        
        # Validate required fields
        if not all([username, email, password, confirm_password]):
            messages.error(request, 'All fields are required', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Check if passwords match
        if password != confirm_password:
            messages.error(request, 'Passwords do not match', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Validate terms acceptance
        if not terms:
            messages.error(request, 'You must accept the terms and conditions', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Validate username length
        if len(username) < 3 or len(username) > 20:
            messages.error(request, 'Username must be between 3 and 20 characters', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Validate username has no spaces
        if ' ' in username:
            messages.error(request, 'Username cannot contain spaces', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Validate username format (alphanumeric and underscores only)
        if not username.replace('_', '').isalnum():
            messages.error(request, 'Username can only contain letters, numbers, and underscores', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
            
        # Check if username already exists (case-insensitive)
        if User.objects.filter(username__iexact=username).exists():
            messages.error(request, 'This username is already taken. Please choose another one.', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Check if email already exists (case-insensitive)
        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, 'This email is already registered. Please use a different email address.', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Validate email format
        from django.core.validators import validate_email
        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, 'Please enter a valid email address', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
        
        # Create new user
        try:
            # Validate password using Django's password validation
            try:
                validate_password(password)
            except ValidationError as e:
                messages.error(request, '\n'.join(e.messages), extra_tags='danger')
                return render(request, 'auth_register_basic.html', context)
            
            # Create the user
            user = User.objects.create_user(
                username=username.strip(),  # Remove any accidental whitespace
                email=email,
                password=password
            )
            
            messages.success(request, 'Registration successful! Please login with your credentials.', extra_tags='success')
            return redirect('auth-login-basic')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}', extra_tags='danger')
            return render(request, 'auth_register_basic.html', context)
    
    return render(request, 'auth_register_basic.html', context)

def login_user(request):
    context = {
        "layout_path": TemplateHelper.set_layout("layout_blank.html", {}),
    }
    
    if request.method == 'POST':
        username = request.POST.get('email-username')
        password = request.POST.get('password')
        remember = request.POST.get('remember', False)
        
        print(f"Login attempt - Username/Email: {username}")  # Debug print
        
        if not all([username, password]):
            messages.error(request, 'Both username/email and password are required')
            return render(request, 'auth_login_basic.html', context)
        
        # First try to authenticate with username
        user = authenticate(request, username=username, password=password)
        
        # If authentication with username fails, try with email
        if user is None:
            try:
                user_obj = User.objects.get(email=username)
                user = authenticate(request, username=user_obj.username, password=password)
                print(f"Email login attempt - Found user: {user_obj.username}")  # Debug print
            except User.DoesNotExist:
                print("Email login attempt - User not found")  # Debug print
                user = None
        
        if user is not None:
            if user.is_active:
                login(request, user)
                print(f"Login successful for user: {user.username}")  # Debug print
                
                # Handle remember me
                if not remember:
                    request.session.set_expiry(0)
                
                # Redirect to next URL if provided, otherwise to dashboard
                next_url = request.POST.get('next') or request.GET.get('next') or 'index'
                return redirect(next_url)
            else:
                messages.error(request, 'Your account is disabled')
        else:
            messages.error(request, 'Invalid email/username or password')
        
        return render(request, 'auth_login_basic.html', context)
    
    return render(request, 'auth_login_basic.html', context)

def logout_user(request):
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('auth-login-basic')

@login_required
def profile_view(request):
    context = {
        "layout_path": TemplateHelper.set_layout("layout_vertical.html", {}),
    }
    return render(request, 'auth_profile.html', context)

@login_required
def change_password(request):
    context = {
        "layout_path": TemplateHelper.set_layout("layout_vertical.html", {}),
    }
    
    if request.method == 'POST':
        current_password = request.POST.get('current_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        
        if not all([current_password, new_password, confirm_password]):
            messages.error(request, 'All fields are required', extra_tags='danger')
            return render(request, 'auth_change_password.html', context)
        
        # Verify current password
        if not request.user.check_password(current_password):
            messages.error(request, 'Current password is incorrect', extra_tags='danger')
            return render(request, 'auth_change_password.html', context)
        
        # Check if new passwords match
        if new_password != confirm_password:
            messages.error(request, 'New passwords do not match', extra_tags='danger')
            return render(request, 'auth_change_password.html', context)
        
        # Validate new password
        try:
            validate_password(new_password, request.user)
        except ValidationError as e:
            messages.error(request, '\n'.join(e.messages), extra_tags='danger')
            return render(request, 'auth_change_password.html', context)
        
        try:
            # Change password
            request.user.set_password(new_password)
            request.user.save()
            
            # Update session to prevent logout
            update_session_auth_hash(request, request.user)
            
            messages.success(request, 'Your password has been successfully changed!', extra_tags='success')
            return redirect('index')  # Redirect to the dashboard/analysis page
            
        except Exception as e:
            messages.error(request, f'Error changing password: {str(e)}', extra_tags='danger')
            return render(request, 'auth_change_password.html', context)
    
    return render(request, 'auth_change_password.html', context)

def forgot_password(request):
    # Initialize context with proper layout path
    context = {
        "layout_path": TemplateHelper.set_layout("layout_blank.html", {}),
    }
    
    if request.method == 'POST':
        email = request.POST.get('email')
        
        if not email:
            messages.error(request, 'Please enter your email address', extra_tags='danger')
            return render(request, 'auth_forgot_password_basic.html', context)
        
        try:
            user = User.objects.get(email=email)
            print(f"Found user: {user.username} with email: {email}")  # Debug print
            
            # Generate password reset token
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            
            # Build password reset link
            reset_link = f"{settings.BASE_URL}/auth/reset-password/{uid}/{token}/"
            print(f"Reset link generated: {reset_link}")  # Debug print
            
            try:
                # Email content
                email_context = {
                    'user': user,
                    'reset_link': reset_link,
                }
                html_message = render_to_string('email/password_reset_email.html', email_context)
                plain_message = strip_tags(html_message)
                
                print("Attempting to send email...")  # Debug print
                print(f"From: {settings.DEFAULT_FROM_EMAIL}")  # Debug print
                print(f"To: {email}")  # Debug print
                
                # Send email
                send_mail(
                    subject='Password Reset Request - FakeGuard',
                    message=plain_message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    html_message=html_message,
                    fail_silently=False,
                )
                
                print("Email sent successfully")  # Debug print
                messages.success(request, 'Password reset instructions have been sent to your email.', extra_tags='success')
                return redirect('auth-login-basic')
                
            except Exception as e:
                print(f"Email sending error: {str(e)}")  # Debug print
                print(f"Email settings: HOST={settings.EMAIL_HOST}, PORT={settings.EMAIL_PORT}, USER={settings.EMAIL_HOST_USER}")  # Debug print
                messages.error(request, f'Error sending email: {str(e)}', extra_tags='danger')
                return render(request, 'auth_forgot_password_basic.html', context)
                
        except User.DoesNotExist:
            print(f"No user found with email: {email}")  # Debug print
            # For security reasons, show the same message even if email doesn't exist
            messages.success(request, 'If an account exists with this email, password reset instructions have been sent.', extra_tags='success')
            return redirect('auth-login-basic')
    
    return render(request, 'auth_forgot_password_basic.html', context)

@login_required
def update_profile(request):
    if request.method == 'POST':
        field = request.POST.get('field')
        
        if field == 'photo':
            if 'photo' not in request.FILES:
                messages.error(request, 'No photo was uploaded')
                return redirect('auth-profile')
                
            photo = request.FILES['photo']
            
            # Validate file type
            if not photo.content_type.startswith('image/'):
                messages.error(request, 'Please upload an image file')
                return redirect('auth-profile')
                
            # Validate file size (5MB)
            if photo.size > 5 * 1024 * 1024:
                messages.error(request, 'Photo size must be less than 5MB')
                return redirect('auth-profile')
                
            try:
                # Update profile photo
                request.user.profile.photo = photo
                request.user.profile.save()
                messages.success(request, 'Profile photo updated successfully')
            except Exception as e:
                messages.error(request, f'Error updating profile photo: {str(e)}')
                
            return redirect('auth-profile')
        
        elif field == 'remove_photo':
            try:
                # Remove the old photo file
                if request.user.profile.photo:
                    request.user.profile.photo.delete()
                request.user.profile.save()
                messages.success(request, 'Profile photo removed successfully')
            except Exception as e:
                messages.error(request, f'Error removing profile photo: {str(e)}')
            return redirect('auth-profile')
            
        value = request.POST.get(field)
        
        if not value:
            messages.error(request, f'{field.title()} cannot be empty')
            return redirect('auth-profile')
            
        if field == 'username':
            # Validate username length
            if len(value) < 3 or len(value) > 20:
                messages.error(request, 'Username must be between 3 and 20 characters')
                return redirect('auth-profile')
                
            # Validate username has no spaces
            if ' ' in value:
                messages.error(request, 'Username cannot contain spaces')
                return redirect('auth-profile')
                
            # Validate username format (alphanumeric and underscores only)
            if not value.replace('_', '').isalnum():
                messages.error(request, 'Username can only contain letters, numbers, and underscores')
                return redirect('auth-profile')
                
            # Check if username is taken
            if User.objects.exclude(pk=request.user.pk).filter(username__iexact=value).exists():
                messages.error(request, 'This username is already taken')
                return redirect('auth-profile')
                
            request.user.username = value.strip()
            messages.success(request, 'Username updated successfully')
            
        elif field == 'email':
            # Validate email format
            from django.core.validators import validate_email
            try:
                validate_email(value)
            except ValidationError:
                messages.error(request, 'Please enter a valid email address')
                return redirect('auth-profile')
                
            # Check if email is taken
            if User.objects.exclude(pk=request.user.pk).filter(email=value).exists():
                messages.error(request, 'This email is already registered')
                return redirect('auth-profile')
                
            request.user.email = value
            messages.success(request, 'Email updated successfully')
            
        try:
            request.user.save()
        except Exception as e:
            messages.error(request, f'Error updating profile: {str(e)}')
            
        return redirect('auth-profile')
        
    return redirect('auth-profile')
