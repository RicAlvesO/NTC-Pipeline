import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualization():

    # Function to check for null values in the dataset
    def plot_missing_values_heatmap(self,base_data):
        plt.figure(figsize=(10, 6))
        sns.heatmap(base_data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
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
    def plot_correlation_matrix(self, base_data):
        # Select numeric columns
        numeric_columns = base_data.select_dtypes(include=['float64', 'int64']).columns
        
        # Check if there are any numeric columns
        if numeric_columns.empty:
            print("No numeric columns to plot correlation matrix.")
            return
        
        # Calculate the correlation matrix
        corr_matrix = base_data[numeric_columns].corr()
        
        # Check if correlation matrix has valid data (non-NaN)
        if corr_matrix.isnull().all().all():
            print("Correlation matrix contains only NaN values.")
            return
        
        # Plot the heatmap if checks pass
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlation Matrix")
        plt.show()

    # Function to plot TCP stream time series analysis
    def plot_tcp_stream_time_series(self,base_data):
        plt.figure(figsize=(14, 6))
        sns.lineplot(x=base_data.index, y='tcp.time_relative', data=base_data, color='teal')
        plt.title("Relative Time of TCP Stream Events")
        plt.xlabel("Index")
        plt.ylabel("TCP Relative Time")
        plt.show()

