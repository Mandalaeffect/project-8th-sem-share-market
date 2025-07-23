import pandas as pd
from django.shortcuts import render
from .forms import UploadFileForm
from django.contrib import messages
from django.http import HttpResponse

from .forecast import run_forecast_on_df


def upload_file(request):
    """
    Handle file upload, run 90-day forecast, and render results page with download option.
    """
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            try:
                # Read uploaded file into dataframe
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Run forecast for next 90 days, without next-day verification (not meaningful for 90 days)
                result_df, plot_url, metrics, test_df, verification = run_forecast_on_df(
                    df,
                    future_days=90,        # 🔷 forecast 90 trading days
                    verify_next_day=False, # 🔷 skip verification (since we can’t verify 90 days in advance)
                    symbol=None
                )

                # Render result table as HTML
                result_html = result_df.to_html(classes='table table-striped', index=False)

                # Save result CSV string to session for later download
                request.session['result_csv'] = result_df.to_csv(index=False)

                return render(request, 'forecasting/result.html', {
                    'result_table': result_html,
                    'plot_url': plot_url,
                    'metrics': metrics,
                    'verification': None  # explicitly pass None since we didn’t verify
                })
            except Exception as e:
                messages.error(request, f"Error processing file: {e}")
    else:
        form = UploadFileForm()

    return render(request, 'forecasting/upload.html', {'form': form})


def download_csv(request):
    """
    Serve the result CSV as a downloadable file.
    """
    csv_content = request.session.get('result_csv')
    if not csv_content:
        return HttpResponse("No result to download. Please upload and process a file first.", status=400)

    response = HttpResponse(csv_content, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="forecast_result.csv"'
    return response
