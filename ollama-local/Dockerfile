FROM ollama/ollama:latest


RUN  apt update -y
RUN apt install curl  wget -y

EXPOSE 11434


COPY ./entrypoint.sh /

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]