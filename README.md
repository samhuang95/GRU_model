# GRU_model
This is for E-commerce to predict the daily KPI.
The model is referenced from TEJ and then fitted with my store sales data.

### Training note
- Model layer
<img decoding="async" src="https://i.imgur.com/1bNOwjg.png" width="30%">

- Loss function 
<img decoding="async" src="https://i.imgur.com/nv7jY6l.png" width="50%">

- Prediction and origin data
<img decoding="async" src="https://i.imgur.com/hmLy2tG.png" width="50%">

- Predict future data
<img decoding="async" src="https://i.imgur.com/AshTR83.png" width="50%">

### Thoughts
To the reader:<br>
It can be adjusted according to your data.<br>
I try to adjust some parameters to fit the data better. <br>
I reduce the DROPOUT parameters, but the result will make the predicted value smooth, so 30% is my finally choose.<br>
I added more neural layer, but it had nothing change, so it is enough for me to evaluate two neural layers.<br>
