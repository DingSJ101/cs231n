FROM python:3.9

RUN pip install jupyter notebook

RUN apt update && apt install -y vim  sudo

RUN addgroup --gid 1000 dsj && \
    adduser --uid 1000 --ingroup dsj --disabled-password dsj &&\
    echo "dsj    ALL=(ALL)    NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace
COPY start.sh /etc/init.d/

CMD ["sh","/etc/init.d/start.sh"]

# # 添加用户：赋予sudo权限，指定密码
# RUN useradd --create-home --no-log-init --shell /bin/bash dsj \
#     && adduser dsj sudo \
#     && echo "dsj:123456" | chpasswd

# # 改变用户的UID和GID
# # RUN usermod -u 1000 ${user} && usermod -G 1000 ${user}

# # 指定容器起来的工作目录
# WORKDIR /home/${user}

# # 指定容器起来的登录用户
# USER ${user}