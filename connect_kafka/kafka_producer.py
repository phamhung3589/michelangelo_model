from kafka import KafkaProducer
from time import sleep
from connect_kafka.constant_cba_message import Constant
from connect_kafka import CbaMessage_pb2


class Producer:

    def __init__(self, message_type, group_id, process_id, session_id, status, contents):
        self.message = CbaMessage_pb2.CbaMessage()
        self.message.messageType = message_type
        self.message.groupID = group_id
        self.message.processID = process_id
        self.message.sessionID = session_id
        self.message.status = status
        for k, v in contents.items():
            self.message.content.__setitem__(k, v)

    def add_data(self):
        """
        add data to object of kafka message (protocol buffer)
        :return: save data to initial variable message
        """

        self.message.createdTime = 1111
        self.message.effectTime = 1234
        self.message.expireTime = 123
        self.message.timeOutResponse = 123
        self.message.timeOutProcess = 123
        self.message.messageNumber = 100
        self.message.description = "testing"

    def log_send_success(self):
        print("sent successful")

    def log_send_error(self):
        print("error")

    def run(self, topic):
        # producer = KafkaProducer(bootstrap_servers=['172.16.28.248:6667'])
        producer = KafkaProducer(bootstrap_servers=['172.16.25.130:9092'])
        self.add_data()
        for e in range(1):
            producer.send(topic, value=self.message.SerializeToString())
            print("sent")
            sleep(5)


if __name__ == "__main__":
    message_type = Constant.message_type_offline_request
    group_id = Constant.group_id_recommender
    process_id = Constant.process_id_recommender_predict
    session_id = "4"
    status = "RUNNING"
    contents = {"user": "84349587967", "num_recommend_item": "5", "evaluation_metric": "ndcg"}
    producer = Producer(message_type, group_id, process_id, session_id, status, contents)
    producer.run("cba_request")
