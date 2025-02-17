using CSV, DataFrames, Statistics, Plots, StatsBase, SkipNan, CategoricalArrays, Dates, StatsModels, GLM

# Function to compute mean and standard deviation of wind direction
function wind_stats(wind_directions)
    wind_radians = deg2rad.(wind_directions)  # Convert to radians

    # Compute mean direction in radians
    s_a = mean(skipnan(sin.(wind_radians)))
    c_a = mean(skipnan(cos.(wind_radians)))
    mean_direction_rad = atan(s_a, c_a)

    # Convert mean direction back to degrees
    mean_direction_deg = rad2deg(mean_direction_rad)
    mean_direction_deg = mean_direction_deg < 0 ? mean_direction_deg + 360 : mean_direction_deg

    # Compute circular standard deviation
    epsilon = sqrt(1 - s_a^2 - c_a^2)
    std_direction = asin(epsilon) * (1 + (2 / sqrt(3) - 1) * epsilon^3)

    # Convert standard deviation back to degrees
    std_direction_deg = rad2deg(std_direction)
    std_direction_deg = std_direction_deg < 0 ? std_direction_deg + 360 : std_direction_deg

    return mean_direction_deg, std_direction_deg
end

function main()
 #NOTE HEIGHTS CAN CHANGE
    #read cleaned file
    lidar_data21 = CSV.read("Cleaned files/modified_lidar_data25_2025-02-17_120327.csv", DataFrame, header = true)

    #DESCRIPTIVE STATISTICS 
    #WIND SPEED Horizontal
    selected_speed_cols = [:"Horizontal Wind Speed (m/s) at 300m", :"Horizontal Wind Speed (m/s) at 278m", :"Horizontal Wind Speed (m/s) at 248m", :"Horizontal Wind Speed (m/s) at 218m", :"Horizontal Wind Speed (m/s) at 188m", :"Horizontal Wind Speed (m/s) at 158m", :"Horizontal Wind Speed (m/s) at 128m", :"Horizontal Wind Speed (m/s) at 98m", :"Horizontal Wind Speed (m/s) at 68m", :"Horizontal Wind Speed (m/s) at 38m", :"Horizontal Wind Speed (m/s) at 10m"] 
    
    # Calculate the mean and standard deviation for each selected column, skipping NaN values
    speed_means = [round(mean(skipnan(lidar_data21[!, col])), digits=2) for col in selected_speed_cols]
    speed_std_devs = [round(std(skipnan(lidar_data21[!, col])), digits=2) for col in selected_speed_cols]
    stats_wind_speeds = DataFrame(Column = selected_speed_cols, Mean = speed_means, Std_Dev = speed_std_devs)
    println(stats_wind_speeds)

     #WIND Direction
     selected_direction_cols = [
        :"Wind Direction (deg) at 300m", 
        :"Wind Direction (deg) at 278m", 
        :"Wind Direction (deg) at 248m", 
        :"Wind Direction (deg) at 218m", 
        :"Wind Direction (deg) at 188m", 
        :"Wind Direction (deg) at 158m", 
        :"Wind Direction (deg) at 128m", 
        :"Wind Direction (deg) at 98m", 
        :"Wind Direction (deg) at 68m", 
        :"Wind Direction (deg) at 38m", 
        :"Wind Direction (deg) at 10m"
    ]
    
    direction_means = [round(wind_stats(skipnan(lidar_data21[!, col]))[1], digits=2) for col in selected_direction_cols]
    direction_std_devs = [round(wind_stats(skipnan(lidar_data21[!, col]))[2], digits=2) for col in selected_direction_cols]
    stats_wind_directions = DataFrame(Column = selected_direction_cols, Mean = direction_means, Std_Dev = direction_std_devs)
    println(stats_wind_directions)

    #TIME SERIES (LOTS OF LOST DATA AT HIGHER HEIGHTS)
    #Wind Speed
    # Adjust the timestamps
    adjusted_timestamps = lidar_data21[:, "Timestamp (s)"] .- 808358407 #Hardcode this minus could definitely not use a magic number
    adjusted_timestamps_hrs = adjusted_timestamps/3600
    # Loop through each column and create a separate plot for each one
    for (i, col) in enumerate(selected_speed_cols)
        p = plot(adjusted_timestamps_hrs, lidar_data21[!, col], 
             xlabel="Time (hrs)", ylabel="Horizontal Wind Speed (m/s)", 
             label="Height: $(selected_speed_cols[i])", linewidth=2)
        p = plot(p, framestyle=:box, grid=:on)
        savefig(p, "output/25_wind_speed_time_$(i).pdf")
        display(p)
    end

    #Mean wind speed vs height plots
    heights = [300, 278, 248, 218, 188, 158, 128, 98, 68, 38, 10]
    wind_speeds = [mean(skipnan(lidar_data21[!, col])) for col in selected_speed_cols]
    p_profile = plot(heights, wind_speeds, xlabel="Height (m)", ylabel="Mean Wind Speed (m/s)", 
                    marker=:o, linewidth=2, label="Wind Speed")
    p_profile = plot!(p_profile, framestyle=:box, grid=:on)
    display(p_profile)

    # Power law fit (wind speed vs height)
    # Log-transform the height and wind speed data for regression
    log_heights = log.(heights)
    log_wind_speeds = log.(wind_speeds)
    # Create a DataFrame with the log-transformed values for regression
    regression_data = DataFrame(log_heights = log_heights, log_wind_speeds = log_wind_speeds)
    # Linear regression to fit the power law
    formula = @formula(log_wind_speeds ~ log_heights)
    # Fit the linear regression model using `lm`
    lm_model = lm(formula, regression_data)
        # Extract the regression coefficients
    intercept = coef(lm_model)[1]  # This is ln(U_0) - α ln(z_0)
    slope = coef(lm_model)[2]      # This is α

    # Calculate the fitted wind speeds using the power law equation: U(z) = U_0 * (z/z_0)^α
    # To do this, first calculate the fitted U_0 from the intercept
    # ln(U_0) = intercept + α * ln(z_0), so U_0 = exp(intercept)
    U_0 = exp(intercept + slope * log(heights[1]))  # Assuming z_0 is the first height

    # Generate the fitted wind speeds from the power law: U(z) = U_0 * (z/z_0)^α
    fitted_wind_speeds = U_0 * (heights ./ heights[1]).^slope

    # Plot the original data (log-log scale)
    p = plot(log.(heights), log.(wind_speeds), label="Observed Data", marker=:o, linewidth=2)
    
    # Plot the fitted power law curve
    plot!(log.(heights), log.(fitted_wind_speeds), label="Fitted Power Law", linewidth=2)

    # Labels and grid
    xlabel!("Log(Height) [log(m)]")
    ylabel!("Log(Wind Speed) [log(m/s)]")
    title!("Power Law Fit: Wind Speed vs Height")
    savefig(p, "output/25_power_law.pdf")
    # Display the plot
    display(p)


    # Calculate hourly means for wind speed at different heights
    hourly_bins = floor.(Int, adjusted_timestamps_hrs / 1)  # 1-hour bins
    hourly_means = [mean(skipnan(lidar_data21[hourly_bins .== b, col])) for b in unique(hourly_bins), col in selected_speed_cols]

    # Plot hourly wind profile
    ph_profile = plot()
    for (i, col) in enumerate(selected_speed_cols)
        # Ensure `hourly_means` is correctly structured as an array of time and wind speed
        plot!(ph_profile, unique(hourly_bins), hourly_means[:, i], label="Height: $(heights[i])", linewidth=2, marker=:o)
    end
  
    xlabel!("Time (hrs)")
    ylabel!("Average Wind Speed (m/s)")
    plot!(ph_profile, legend=:bottom)
    savefig(ph_profile, "output/25_hourly_average_wind.pdf")
    display(ph_profile)
  
  
end
main()