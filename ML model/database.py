import datetime


class DataBase:
    def __init__(self, filename):
        self.filename = filename
        self.users = None
        self.file = None
        self.load()

    def load(self):
        self.file = open(self.filename, "r")
        self.users = {}

        for line in self.file:
            user_id,date = line.split(";")
            self.users[user_id] = (date)

        self.file.close()

    def get_user(self, user_id):
        self.file=open("data.txt","r")
        if user_id in self.users:
            return self.users[user_id][0]
        else:
            return -1

    def add_user(self,user_id,date):
        self.users[user_id] = (date)
        self.save()
        return 1


    def validate(self, email, password):
        if self.get_user(email) != -1:
            return self.users[email][0] == password
        else:
            return False

    def save(self):
        with open(self.filename, "a") as f:
            for user in self.users:
                f.write(user + ";" + self.users[user][0] + "\n")

