# Stock & FX Prediction Dashboard - Deployment Notes

## ⚠️ Important: API Keys
- The `.env` file contains your Alpha Vantage API key. **Do NOT commit this to public repositories.**
- For public deployments, set the API key as an environment variable in the deployment provider's dashboard.

## How to Deploy
1. Deploy to Streamlit Community Cloud (recommended for Streamlit apps):
   - Go to https://share.streamlit.io/
   - Connect your GitHub repo, set the main file as `app.py`.
   - Set the Alpha Vantage API key as a secret in the Streamlit Cloud settings.
2. Or deploy to Netlify using the included `netlify.toml` (requires custom setup for Streamlit, not always supported for Python backends).

## Security
- `.env` and all secrets are gitignored by default.
- Never expose private API keys in public code or client-side JS.

## Sharing
- After deployment, share the public URL with anyone. They do not need to install anything locally.
