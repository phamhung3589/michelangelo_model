from kafka import KafkaConsumer
from connect_kafka.constant_cba_message import Constant
from connect_kafka import CbaMessage_pb2
from recommendation.model.ALS import ModelALS
from connect_kafka.kafka_producer import Producer
import ast


class Consumer:

    def __init__(self):
        self.cba_message = CbaMessage_pb2.CbaMessage()

    def run(self):

        consumer = KafkaConsumer(Constant.topic_request,
                                 # bootstrap_servers=['172.16.28.250:6667'],
                                 bootstrap_servers=['172.16.25.130:9092'],
                                 auto_offset_reset='latest',
                                 group_id = 'test_cba',
                                 enable_auto_commit=True)

        for message in consumer:
            self.cba_message.ParseFromString(message.value)
            ALS_model = ModelALS()

            if self.cba_message.messageType == Constant.message_type_offline_request and \
               self.cba_message.groupID == Constant.group_id_recommender and \
               self.cba_message.processID == Constant.process_id_recommender_training:

                print("training")

                # Train model
                spark_config = ALS_model.create_spark_environment()
                ALS_model.train(num_recommend_item=5, evaluation_metric='ndcg',sc=spark_config)
                print("training model has been done")

                spark_config.stop()

            elif self.cba_message.messageType == Constant.message_type_offline_request and \
                 self.cba_message.groupID == Constant.group_id_recommender and \
                 self.cba_message.processID == Constant.process_id_recommender_predict:

                Consumer.send_message_to_topic({}, self.cba_message.sessionID, Constant.status_accepted,
                                               Constant.topic_process_status)

                spark_config = ALS_model.create_spark_environment()
                print("load environment done")

                list_user = ast.literal_eval(self.cba_message.content["list_msisdn"])
                num_item_recommend = int(self.cba_message.content["num_recommend_item"])
                # Load model
                ALS_model.load_model(spark_config)
                print("load model done")
                Consumer.send_message_to_topic({}, self.cba_message.sessionID, Constant.status_running,
                                               Constant.topic_process_status)

                # Recommend offers for list user
                list_offer_predict = ALS_model.predict_list(list_user=list_user, num_item_recommend=num_item_recommend,
                                                            spark_config=spark_config)

                Consumer.send_message_to_topic(list_offer_predict, self.cba_message.sessionID, Constant.status_finished,
                                               Constant.topic_process_status)
                Consumer.send_message_to_topic(list_offer_predict, self.cba_message.sessionID, Constant.status_finished,
                                               Constant.topic_response)

                spark_config.stop()

    @staticmethod
    def send_message_to_topic(list_offer_predict, sessionID, status, topic):
        message_type = Constant.message_type_offline_response
        group_id = Constant.group_id_recommender
        process_id = Constant.process_id_recommender_predict
        contents = {"table_recommendation_item": str(list_offer_predict)}
        producer = Producer(message_type, group_id, process_id, sessionID, status, contents)
        producer.run(topic)


if __name__ == "__main__":
    consumer = Consumer()
    consumer.run()
