# Google OAuth Setup Guide

## Overview
This application now includes Google OAuth authentication that restricts access to users with `@uni.minerva.edu` or `@minerva.edu` email addresses, just like the Jupyter notebook.

## Required Steps

### 1. Create Google OAuth Credentials

1. **Go to [Google Cloud Console](https://console.cloud.google.com/)**
2. **Create a new project** (or select existing one)
3. **Enable Google APIs:**
   - Go to "APIs & Services" > "Library"
   - Search for and enable "Google+ API" or "Google Identity API"
4. **Create OAuth 2.0 Credentials:**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Web application"
   - Set **Authorized redirect URIs** to:
     - `http://localhost:3000/callback` (for development)
     - `https://your-domain.com/callback` (for production)
   - Copy the **Client ID** and **Client Secret**

### 2. Configure Environment Variables

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file with your credentials:**
   ```
   GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   SECRET_KEY=your-secret-key-change-in-production
   OPENAI_API_KEY=your-openai-api-key
   ```

### 3. Install Dependencies

Dependencies are already installed:
- `authlib` - OAuth library
- `requests` - HTTP requests

### 4. Run the Application

```bash
python3 app.py
```

The app will run on `http://localhost:3000`

## How Authentication Works

### 1. **Login Flow:**
- Users visit the app and are redirected to `/login`
- Clicking "Sign in with Google" starts OAuth flow
- Google authenticates user and returns to `/callback`
- App checks if email ends with `@uni.minerva.edu` or `@minerva.edu`
- If valid, user is logged in and redirected to main app
- If invalid, user sees error and stays on login page

### 2. **Protected Routes:**
- All main application routes require authentication
- API endpoints (`/api/transcribe`, `/api/ai-chat`, etc.) are protected
- Unauthenticated users are redirected to login page

### 3. **Session Management:**
- User info stored in Flask session
- Session expires when browser closes
- Logout clears session and redirects to login

### 4. **User Interface:**
- Header shows user's name, email, and profile picture
- Logout button in top right
- Flash messages show login status

## Security Features

✅ **Domain Restriction:** Only `@uni.minerva.edu` and `@minerva.edu` emails allowed  
✅ **Session Protection:** All sensitive routes require authentication  
✅ **OAuth Security:** Uses Google's secure OAuth 2.0 flow  
✅ **HTTPS Ready:** Works with HTTPS in production  

## Files Modified

- `app.py` - Added OAuth routes, authentication decorators, session management
- `templates/index.html` - Added user info display and logout button
- `templates/login.html` - New login page with Google sign-in
- `static/css/style.css` - Added styles for user info and login page
- `.env.example` - Template for environment variables

## Troubleshooting

### Common Issues:

1. **"OAuth client not found"**
   - Check GOOGLE_CLIENT_ID in .env
   - Verify redirect URI in Google Console

2. **"Access restricted" error**
   - Ensure you're signing in with @uni.minerva.edu or @minerva.edu email
   - Check ALLOWED_DOMAINS in app.py

3. **"Authentication failed"**
   - Check GOOGLE_CLIENT_SECRET in .env
   - Ensure Google+ API is enabled in Google Console

4. **Session issues**
   - Set a secure SECRET_KEY in .env
   - Clear browser cookies and try again

## Production Deployment

When deploying to production:

1. **Update redirect URI** in Google Console to your production domain
2. **Set secure SECRET_KEY** (use `python3 -c "import secrets; print(secrets.token_hex(16))"`)
3. **Use HTTPS** for secure OAuth flow
4. **Set environment variables** on your hosting platform