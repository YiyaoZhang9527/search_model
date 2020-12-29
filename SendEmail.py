import smtplib
from email.mime.text import MIMEText


default_key="" #邮箱授权码

def send_email(receiverMail="" #接受者邮件地址
             ,subject = "自动邮件测试" #邮件主题
             ,content = "hello dear" #邮件内容
             ,authCode = default_key
             ,senderMail = '' #发送者地址邮箱
            ):
    msg = MIMEText(content,"plain","utf-8")
    msg['Subject'] = subject
    msg['From'] = senderMail
    msg['To'] = receiverMail
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
        print('成功连接到邮件服务器')
        server.login(senderMail, authCode)
        print('成功登录邮箱')
        server.sendmail(senderMail, receiverMail, msg.as_string())
        print('邮件发送成功')
    except smtplib.SMTPException as e:
        print('邮件发送异常',e)
    finally:
        server.quit()
        
        


