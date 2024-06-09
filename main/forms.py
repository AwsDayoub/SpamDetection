from django import forms

class TrainForm(forms.Form):
    data_file = forms.FileField()

class PredictForm(forms.Form):
    email_text = forms.CharField(widget=forms.Textarea)
    k_value = forms.IntegerField(min_value=1)
