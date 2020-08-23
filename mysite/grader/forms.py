from django import forms

from .models import Essay


class AnswerForm(forms.ModelForm):
    answer = forms.CharField(max_length=100000, widget=forms.Textarea(
        attrs={'rows': 20, 'placeholder':  "Your essay here"}))

    class Meta:
        model = Essay
        fields = ['answer']
