import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import json
import requests
from mlrun.execution import MLClientCtx
from typing import List


def slack_notify(
    context: MLClientCtx,
    webhook_url: str = "URL",
    slack_blocks: List[str] = [],
    notification_text: str = "Notification",
) -> None:
    """Summarize a table
    :param context:         the function context
    :param webhook_url:     Slack incoming webhook URL. Please read: https://api.slack.com/messaging/webhooks
    :param notification_text:            Notification text
    :param slack_blocks:          Message blocks list. NOT IMPLEMENTED YET
    """

    data = {"text": notification_text}
    print("====", webhook_url)
    response = requests.post(
        webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    )

    print("Response: " + str(response.text))
    print("Response code: " + str(response.status_code))
