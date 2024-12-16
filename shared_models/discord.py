import requests,time

class Discord:
    @staticmethod
    def send_discord_webhook(message, nom="[NAME]", direction="LONG", harmonic="Gartley", owner="auto", color=0, current=0, additional="info", id="", threadid=0):
        webhook_url = "https://discord.com/api/webhooks/1218117882274119780/MjTN_zR-w8VPBlGrAATMHTGxpONdrtlwGWv0imbshyod0Fb5Kb4--Q0krZkj4BQMdl_V"

        if threadid > 0:
            if threadid == 1:  # Enter in trade
                webhook_url = webhook_url + "?thread_id=1304096750327435265"
            if threadid == 2:  # Near PRZ
                webhook_url = webhook_url + "?thread_id=1304096433128996904"
            if threadid == 3:  # True fib
                webhook_url = webhook_url + "?thread_id=1304096400384200737"
            if threadid == 4:  # Exit PRZ
                webhook_url = webhook_url + "?thread_id=1304096651614359567"
            if threadid == 5:  # Exit trade
                webhook_url = webhook_url + "?thread_id=1310870197275594772"
            if threadid == 0:  # Debug
                webhook_url = webhook_url + "?thread_id=1317121312765906988"

        data = {
            "embeds": [{
                "author": {"name": owner},
                "title": "[" + direction + "][" + nom + "] " + harmonic + "",
                "url": "https://grand-guru-429507-r5.ew.r.appspot.com/#"+id,
                "color": color,
                "footer": {
                    "text": "Current price: " + str(current) + "\n" + str(additional)
                },
                "description": message
            }]
        }

        response = requests.post(webhook_url, json=data)

        if response.status_code != 204:
            time.sleep(1)
            response = requests.post(webhook_url, json=data)

