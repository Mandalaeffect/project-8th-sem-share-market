from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Upload CSV or Excel file')
