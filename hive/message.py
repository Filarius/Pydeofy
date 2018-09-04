class MessageType():
    @property
    def Data(self):
        return 0

    @Data.setter
    def Data(self, value):
        pass

    @property
    def EOF(self):
        return 1

    @EOF.setter
    def EOF(self,value):
        pass



class Message:
    def __init__(self,id,type=None,body=None):
        self.id = id
        self.type = type
        self.body = body






