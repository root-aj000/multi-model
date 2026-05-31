"""
Email templates for invitation emails.

Each template function returns an HTML string. You can customize these
templates to match your brand — change colors, logos, layout, wording, etc.

Available template variables:
    {{inviter_name}}  — Display name of the person who sent the invite
    {{tenant_name}}   — Name of the workspace the user is invited to
    {{invite_link}}   — Full URL with token for accepting the invite
    {{expiry_days}}   — Number of days until the invite expires
    {{app_name}}      — Application name (from APP_NAME env var, default "Multi-Model")
"""

import os


def _get_app_name() -> str:
    """Get the application name from env or default."""
    return os.getenv("APP_NAME", "Multi-Model")


def _replace_vars(template: str, **kwargs: str) -> str:
    """Replace {{variable}} placeholders in the template string."""
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


# ---------------------------------------------------------------------------
# Invite Email Template
# ---------------------------------------------------------------------------

INVITE_EMAIL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>You're Invited!</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f5f5f5; font-family: Arial, sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f5f5f5; padding: 40px 0;">
        <tr>
            <td align="center">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08);">

                    <!-- Header Banner -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #ff6b35 0%, #ff8f5e 100%); padding: 40px 40px 30px 40px; text-align: center;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">
                                You're Invited!
                            </h1>
                            <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.85); font-size: 16px;">
                                Join {{tenant_name}} on {{app_name}}
                            </p>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding: 40px;">
                            <p style="margin: 0 0 20px 0; font-size: 16px; color: #333333; line-height: 1.6;">
                                Hi there!
                            </p>
                            <p style="margin: 0 0 20px 0; font-size: 16px; color: #333333; line-height: 1.6;">
                                <strong>{{inviter_name}}</strong> has invited you to join
                                <strong>{{tenant_name}}</strong> on {{app_name}}.
                            </p>
                            <p style="margin: 0 0 30px 0; font-size: 14px; color: #666666; line-height: 1.6;">
                                Click the button below to accept your invitation and create your account:
                            </p>

                            <!-- CTA Button -->
                            <table role="presentation" cellspacing="0" cellpadding="0" style="margin: 0 auto;">
                                <tr>
                                    <td style="border-radius: 12px; background: linear-gradient(135deg, #ff6b35 0%, #ff8f5e 100%);">
                                        <a href="{{invite_link}}"
                                           style="display: inline-block; padding: 16px 40px; color: #ffffff;
                                                  text-decoration: none; font-size: 16px; font-weight: 700;
                                                  border-radius: 12px;">
                                            Accept Invitation
                                        </a>
                                    </td>
                                </tr>
                            </table>

                            <!-- Fallback Link -->
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-top: 30px;">
                                <tr>
                                    <td style="padding: 16px; background-color: #f9f9f9; border-radius: 8px; border: 1px dashed #dddddd;">
                                        <p style="margin: 0 0 8px 0; font-size: 12px; color: #999999;">
                                            If the button doesn't work, copy and paste this link into your browser:
                                        </p>
                                        <p style="margin: 0; font-size: 13px; word-break: break-all;">
                                            <a href="{{invite_link}}" style="color: #ff6b35; text-decoration: none;">{{invite_link}}</a>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; background-color: #fafafa; border-top: 1px solid #eeeeee;">
                            <p style="margin: 0 0 8px 0; font-size: 12px; color: #999999; line-height: 1.5;">
                                This invitation expires in <strong>{{expiry_days}} days</strong>.
                                If you did not expect this invitation, you can safely ignore this email.
                            </p>
                            <p style="margin: 0; font-size: 12px; color: #bbbbbb;">
                                &copy; {{app_name}} &mdash; Powered by <a href="https://github.com" style="color: #bbbbbb; text-decoration: none;">Multi-Model Classification</a>
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def render_invite_email(
    inviter_name: str,
    tenant_name: str,
    invite_link: str,
    expiry_days: int = 7,
) -> str:
    """
    Render the invite email HTML template with the given variables.

    Args:
        inviter_name: Display name of the person who sent the invite.
        tenant_name: Name of the workspace.
        invite_link: Full URL with token for accepting the invite.
        expiry_days: Number of days until the invite expires.

    Returns:
        HTML string with all template variables replaced.
    """
    return _replace_vars(
        INVITE_EMAIL_HTML,
        inviter_name=inviter_name,
        tenant_name=tenant_name,
        invite_link=invite_link,
        expiry_days=str(expiry_days),
        app_name=_get_app_name(),
    )
