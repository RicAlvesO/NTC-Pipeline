import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display, HTML
from matplotlib_venn import venn3

class Visualization():

    # Function to check for null values in the dataset
    def plot_missing_values_heatmap(self, base_data):
        # Convert placeholders to NaN if necessary
        base_data.replace("", np.nan, inplace=True)
        base_data.replace("null", np.nan, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.heatmap(base_data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.show()

    # Function to plot the top 20 source IP addresses
    def plot_top_ip_addresses(self,base_data):
        plt.figure(figsize=(14, 6))
        base_data['ip.src'].value_counts().head(20).plot(kind='bar', color='lightgreen')
        plt.title("Top 20 Source IP Addresses")
        plt.xlabel("Source IP")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Function to plot the top 20 destination IP addresses
    def plot_top_ip_dest_addresses(self,base_data):
        plt.figure(figsize=(14, 6))
        base_data['ip.dst'].value_counts().head(20).plot(kind='bar', color='lightgreen')
        plt.title("Top 20 Destination IP Addresses")
        plt.xlabel("Destination IP")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Function to plot distribution of TCP source port counts
    def plot_tcp_source_ports(self,base_data):
        plt.figure(figsize=(14, 6))
        base_data['tcp.srcport'].value_counts().head(20).plot(kind='bar', color='skyblue')
        plt.title("Top 20 Source TCP Ports")
        plt.xlabel("TCP Source Port")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Function to plot distribution of UDP source port counts
    def plot_udp_source_ports(self,base_data):
        plt.figure(figsize=(14, 6))
        base_data['udp.srcport'].value_counts().head(20).plot(kind='bar', color='salmon')
        plt.title("Top 20 Source UDP Ports")
        plt.xlabel("UDP Source Port")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Function to plot distribution of TCP flags
    def plot_tcp_flags_distribution(self,base_data):
        plt.figure(figsize=(14, 6))
        sns.countplot(x='tcp.flags_str', data=base_data, order=base_data['tcp.flags_str'].value_counts().index, palette='coolwarm')
        plt.title("Distribution of TCP Flags")
        plt.xlabel("TCP Flag")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Function to plot top 10 DNS query names
    def plot_dns_queries(self,base_data):
        plt.figure(figsize=(14, 6))
        base_data['dns.qry_name'].value_counts().head(10).plot(kind='bar', color='mediumpurple')
        plt.title("Top 10 DNS Query Names")
        plt.xlabel("DNS Query Name")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    

    # Function to plot correlation matrix for numeric columns
    def plot_correlation_matrix(self, base_data, corr_threshold=0.5):
        # Select numeric columns
        numeric_columns = base_data.select_dtypes(include=['float64', 'int64']).columns
        
        # Check if there are any numeric columns
        if numeric_columns.empty:
            print("No numeric columns to plot correlation matrix.")
            return
        
        # Calculate the correlation matrix
        corr_matrix = base_data[numeric_columns].corr()
        
        # Apply threshold to filter only strong correlations
        mask = (corr_matrix >= corr_threshold) | (corr_matrix <= -corr_threshold)
        filtered_corr_matrix = corr_matrix.where(mask, other=0)

        # Plot the heatmap with adjustments
        plt.figure(figsize=(20, 16))
        sns.heatmap(filtered_corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True, 
                    cbar_kws={"shrink": 0.7}, linewidths=0.5)

        plt.title("Correlation Matrix (Thresholded)", fontsize=18)
        plt.xticks(rotation=90, ha='center', fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()

    # Function to plot TCP stream time series analysis
    def plot_tcp_stream_time_series(self,base_data):
        plt.figure(figsize=(14, 6))
        sns.lineplot(x=base_data.index, y='tcp.time_relative', data=base_data, color='teal')
        plt.title("Relative Time of TCP Stream Events")
        plt.xlabel("Index")
        plt.ylabel("TCP Relative Time")
        plt.show()


    def plot_distribution(self, base_data):
        """Plot distribution for a numerical column."""

        for column in base_data.select_dtypes(include=['float64', 'int64']).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(base_data[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()


    # Function to filter columns by missing values and display the desired outputs
    def filter_columns_by_missing_values(self, base_data, threshold):
        # Calculate the percentage of missing values for each column
        missing_percentage = base_data.isnull().mean()

        # Get columns with 0% missing values
        zero_missing = missing_percentage[missing_percentage == 0]
        zero_missing_list = zero_missing.index.tolist()

        # Get columns with missing values less than the threshold and greater than 0
        valid_greater_than_zero_columns = missing_percentage[(missing_percentage < threshold) & (missing_percentage > 0)]
        valid_columns = missing_percentage[(missing_percentage < threshold)]        

        # Get columns with more than 85% missing values
        high_missing = missing_percentage[missing_percentage > threshold]
        high_missing_list = high_missing.index.tolist()

        # Create a DataFrame to display columns with less than 85% missing (excluding 0% missing)
        valid_columns_df = pd.DataFrame({
            'Column': valid_greater_than_zero_columns.index,
            'Missing Percentage': valid_greater_than_zero_columns.values * 100  # Convert to percentage
        })

        # Display the list and table
        print("Columns with 0% Missing Values:", zero_missing_list)
        print(f"Columns with >{threshold} Missing Values:", high_missing_list)

        # Display the table with columns that have missing values less than 85% (excluding 0% missing)
        html_table = valid_columns_df.to_html(index=False, escape=False)
        display(HTML(f"<h3>Columns with <{threshold}% Missing Values (excluding 0%)</h3>{html_table}"))

        # Return the cleaned DataFrame with only valid columns
        base_data_cleaned = base_data[valid_columns.index]

        return base_data_cleaned
    

    def calculate_missing_percentages(self, base_data):
        return base_data.isnull().mean()
    

    def print_zero_missing_columns(self, missing_percentage):
        zero_missing = missing_percentage[missing_percentage == 0].index.tolist()
        print("Columns with 0% Missing Values:", zero_missing)


    def print_high_missing_columns(self, missing_percentage, threshold):
        high_missing = missing_percentage[missing_percentage > threshold].index.tolist()
        print(f"Columns with >{threshold * 100}% Missing Values:", high_missing)


    def print_valid_columns_table(self, missing_percentage, threshold):
        valid_greater_than_zero_columns = missing_percentage[(missing_percentage < threshold) & (missing_percentage > 0)]
        valid_columns_df = pd.DataFrame({
            'Column': valid_greater_than_zero_columns.index,
            'Missing Percentage': valid_greater_than_zero_columns.values * 100  # Convert to percentage
        })
        html_table = valid_columns_df.to_html(index=False, escape=False)
        display(HTML(f"<h3>Columns with <{threshold * 100}% Missing Values (excluding 0%)</h3>{html_table}"))

    
    def filter_valid_columns(self, base_data, missing_percentage, threshold):
        valid_columns = missing_percentage[missing_percentage < threshold].index
        return base_data[valid_columns]
    

    def get_types(self, base_data):
        missing_percentage = self.calculate_missing_percentages(base_data)
        # filtered_columns = [col for col in base_data.columns if col.endswith('.') and col.count('.') == 1]
        filtered_columns = [col for col in base_data.columns if col.startswith('layer_') and col.count('.') == 1]
        filtered_missing_percentage = missing_percentage[filtered_columns]
        available_data_percentage = (1 - filtered_missing_percentage) * 100

        plt.figure(figsize=(10, 6))
        available_data_percentage.sort_values().plot(kind='bar', color='skyblue')
        plt.title('Percentage of Available Data', fontsize=16)
        plt.xlabel('Columns', fontsize=12)
        plt.ylabel('Available Data (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def column_info(self, base_data):

        # Sort the columns alphabetically
        sorted_columns = sorted(base_data.columns)

        # Create a DataFrame with sorted columns and their data types
        sorted_dtypes = base_data[sorted_columns].dtypes.to_frame()

        # Rename the DataFrame's columns for clarity
        sorted_dtypes.columns = ["Data Type"]

        # Display the sorted DataFrame as HTML
        return display(HTML(sorted_dtypes.to_html(index=True)))
    

    def first_x_rows(self, base_data, x):
        return display(HTML(base_data.head(x).to_html()))
    

    def plot_dataset_counts(self, data):
        """Plots the count of documents by dataset."""

        dataset_df = pd.DataFrame(data)  # Convert to DataFrame if it's not already

        # Count occurrences of each dataset
        dataset_counts = dataset_df['dataset'].value_counts().reset_index()
        dataset_counts.columns = ['Dataset', 'Count']

        plt.figure(figsize=(12, 6))
        plt.bar(dataset_counts['Dataset'], dataset_counts['Count'], color='skyblue')
        plt.title('Count of Documents by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def plot_label_counts(self, data):

        """Plots the count of documents by label."""
        label_df = pd.DataFrame(data)  # Convert to DataFrame if it's not already

        # Count occurrences of each label
        label_counts = label_df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']

        # plt.figure(figsize=(12, 6))
        # plt.bar(label_counts['Label'], label_counts['Count'], color='salmon')
        # plt.title('Count of Documents by Label')
        # plt.xlabel('Label')
        # plt.ylabel('Count')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()

        plt.figure(figsize=(4, 4)) 
        plt.pie(label_counts['Count'], labels=label_counts['Label'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title('Count of Documents by Label')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    

    def find_common_columns_from_datasets(self, df):
        """
        Finds common columns across different datasets in a DataFrame,
        excluding columns that are completely empty in each dataset.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with a 'dataset' column.

        Returns:
        list: A list of common column names across all datasets.
        """
        # Get unique datasets
        unique_datasets = df['dataset'].unique()

        # Initialize a dictionary to hold the DataFrames for each dataset
        dataset_dfs = {dataset: df[df['dataset'] == dataset].drop(columns=['dataset']) for dataset in unique_datasets}

        # Initialize a set to hold the common columns
        common_columns = None

        # Iterate through each dataset DataFrame
        for dataset, dataset_df in dataset_dfs.items():
            # Identify columns that are completely empty
            empty_columns = dataset_df.columns[dataset_df.isnull().all()].tolist()

            # Drop empty columns from the current dataset DataFrame
            dataset_df = dataset_df.drop(columns=empty_columns)

            # Update common columns
            if common_columns is None:
                common_columns = set(dataset_df.columns)
            else:
                common_columns.intersection_update(dataset_df.columns)

        # Convert the set to a list for easier reading
        return list(common_columns)

    def check_udp_port_columns(self, base_data):
        # Filter rows where 'udp.port' is not null
        non_null_udp_port = base_data[['udp.port', 'udp.srcport', 'udp.dstport']].dropna(subset=['udp.port'])
        
        # Add comparison columns to check if 'udp.port' matches 'udp.srcport' or 'udp.dstport'
        non_null_udp_port['matches_srcport'] = non_null_udp_port['udp.port'] == non_null_udp_port['udp.srcport']
        non_null_udp_port['matches_dstport'] = non_null_udp_port['udp.port'] == non_null_udp_port['udp.dstport']
        
        # check if there are any rows where 'udp.port' does not match 'udp.srcport' or 'udp.dstport'
        print("Rows where 'udp.port' does not match 'udp.srcport' or 'udp.dstport':")
        if (non_null_udp_port[(non_null_udp_port['matches_srcport'] == False) & (non_null_udp_port['matches_dstport'] == False)]).empty:
            print("No rows found")
        else:
            return(non_null_udp_port[(non_null_udp_port['matches_srcport'] == False) & (non_null_udp_port['matches_dstport'] == False)])


    def check_tcp_port_columns(self, base_data):
        # Filter rows where 'tcp.port' is not null
        non_null_tcp_port = base_data[['tcp.port', 'tcp.srcport', 'tcp.dstport']].dropna(subset=['tcp.port'])
        
        # Add comparison columns to check if 'tcp.port' matches 'tcp.srcport' or 'tcp.dstport'
        non_null_tcp_port['matches_srcport'] = non_null_tcp_port['tcp.port'] == non_null_tcp_port['tcp.srcport']
        non_null_tcp_port['matches_dstport'] = non_null_tcp_port['tcp.port'] == non_null_tcp_port['tcp.dstport']
        
        # check if there are any rows where 'tcp.port' does not match 'tcp.srcport' or 'tcp.dstport'
        print("Rows where 'tcp.port' does not match 'tcp.srcport' or 'tcp.dstport':")
        if (non_null_tcp_port[(non_null_tcp_port['matches_srcport'] == False) & (non_null_tcp_port['matches_dstport'] == False)]).empty:
            print("No rows found")
        else:
            return(non_null_tcp_port[(non_null_tcp_port['matches_srcport'] == False) & (non_null_tcp_port['matches_dstport'] == False)])
    

    def plot_size_distribution(self, base_data):
        # Ensure that 'size' column is not null and is numeric
        non_null_size = base_data['size'].dropna()  # Remove NaN values

        plt.figure(figsize=(10, 6))
        sns.histplot(non_null_size, kde=True, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Size', fontsize=16)
        plt.xlabel('Size', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.show()