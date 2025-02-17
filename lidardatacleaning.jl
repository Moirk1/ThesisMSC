using CSV, DataFrames, Dates

function main() 
    # Read CSV file
    lidar_data21 = CSV.read("Raw files/Wind_2502@Y2025_M01_D25eh (1).csv", DataFrame, header=true)
    #Hard coded so need to check each time
    # List of boolean columns and their corresponding preceding columns HARD CODED
    boolean_columns = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    preceding_columns = [
    [21, 22, 23], [25, 26, 27], [29, 30, 31], 
    [33, 34, 35], [37, 38, 39], [41, 42, 43], [45, 46, 47],
    [49, 50, 51], [53, 54, 55], [57, 58, 59], [61, 62, 63]
]

    # Loop through the boolean columns and convert string-based booleans to actual booleans
    #NEEDED FOR FIRST DATASET  (This probably needs to be checked each time)
    for col in boolean_columns
        # Check if the column is of String7 type (or String)
        if eltype(lidar_data21[!, col]) == String7 || eltype(lidar_data21[!, col]) == String
            # Convert string-based booleans to actual Bool
            lidar_data21[!, col] .= lowercase.(lidar_data21[!, col]) .== "true"
        end
    end
  
  
    # Iterate through each row to use nans
    for i in 1:nrow(lidar_data21)
        for (index, bool_col) in enumerate(boolean_columns)
            if lidar_data21[i, bool_col] == false  # If a boolean column is false
                for col in preceding_columns[index]  # Get corresponding preceding columns
                    lidar_data21[i, col] = NaN  # Or `missing` if preferred
                end
            end
        end
    end

    println("Processing complete!")

    # Save the modified file with a timestamp into cleaned files folder
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    #CHANGE NAMES
    CSV.write("Cleaned files/modified_lidar_data25_$timestamp.csv", lidar_data21)
end

main()
