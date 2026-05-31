"""
Email sending utility for invitation emails.

Uses standard SMTP (configurable via env vars) to send invitation emails.
When SMTP is not configured, logs the invite link instead (development mode).

Email templates are defined in email_templates.py for easy customization.
"""

import logging
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from lib.auth.email_templates import render_invite_email

logger = logging.getLogger(__name__)


def is_smtp_configured() -> bool:
    """Check if SMTP credentials are configured."""
    return all(os.getenv(k) for k in ["SMTP_HOST", "SMTP_USER", "SMTP_PASS"])


def send_invite_email(
    to_email: str,
    invite_link: str,
    tenant_name: str,
    inviter_name: str,
    expiry_days: int = 7,
) -> None:
    """
    Send an invitation email to the specified address.

    When SMTP is not configured, logs the invite link instead (for development).

    Args:
        to_email: Recipient email address.
        invite_link: Full URL with token for accepting the invite.
        tenant_name: Name of the tenant the user is invited to.
        inviter_name: Display name of the person who sent the invite.
        expiry_days: Number of days until the invite expires (default 7).
    """
    if not is_smtp_configured():
        logger.warning(
            "SMTP not configured. Invite link for %s: %s",
            to_email, invite_link,
        )
        return

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)
    smtp_from_name = os.getenv("SMTP_FROM_NAME", "")

    # Render the email template
    html = render_invite_email(
        inviter_name=inviter_name,
        tenant_name=tenant_name,
        invite_link=invite_link,
        expiry_days=expiry_days,
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"You're invited to join {tenant_name}"
    # Use custom display name if configured, e.g. "Multi-Model <noreply@example.com>"
    if smtp_from_name:
        msg["From"] = f"{smtp_from_name} <{smtp_from}>"
    else:
        msg["From"] = smtp_from
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        import smtplib

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info("Invite email sent to %s", to_email)
    except Exception as exc:
        logger.error("Failed to send invite email to %s: %s", to_email, exc)
        # Re-raise so the caller can handle it
        raise
