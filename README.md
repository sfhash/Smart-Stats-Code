# Smart-Stats-Code


Step 1: Extract .input2.zip' and 'input_for_code2/results_smart_stats.zip' files to get input2.csv and results_smart_stats.csv.I zipped them to make them smaller in memory size.<br>

Step 2: Run smart_stats 3.ipynb<br>

Step 3: code2_Plot_results and top20.ipynb<br>


smart_stats 3.ipynb : <br>
   -This is the main file which generates the iutput tables.<br>
   -Input required is only input2.csv. <br>
   -Library Required: from sklearn.ensemble import RandomForestRegressor.<br>
  Output files generated are the following:<br>
 1) results_smart_stats.csv
 2) smart_stats_by_batsman.csv
 3) smart_stats_by_bowler.csv
 4) data_rf_runs_per_ball.csv
 5) impact_score_batsman.csv
 6) impact_score_bowler.csv

code2_Plot_results and top20.ipynb : <br>
   -This notebook contains plots based on the output files of smart_stats 3.ipynb.<br>
   -Input Required is : All the files inside 'input_for_code2' Folder. <br>
  Output files generated are the following:<br>
1) top20_smart_bat_avg.csv
2) top20_smart_econ_rate.csv
3) top20_smart_bowl_avg.csv


