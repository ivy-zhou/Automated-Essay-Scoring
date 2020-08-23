from django.db import models

# Create your models here.


# class Question(models.Model):
#     """ A model of the essay question. """
#     content = models.TextField(max_length=100000)

#     def __str__(self):
#         return str(self.content[0:30])


class Essay(models.Model):
    """ Essay to be submitted. """
    # question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField(max_length=100000)
    score = models.IntegerField(null=True, blank=True)
    semantic = models.IntegerField(null=True, blank=True)
    tense = models.IntegerField(null=True, blank=True)
    accuracy = models.IntegerField(null=True, blank=True)
    wordcount = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return str(self.content[0:30])
