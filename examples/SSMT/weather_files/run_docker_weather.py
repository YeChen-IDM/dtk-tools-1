from simtools.AssetManager.FileList import FileList
from simtools.Managers.WorkItemManager import WorkItemManager
from simtools.SetupParser import SetupParser


def main():
    wi_name = 'ERA5 weather generation'

    # A .csv or a demographics file containing input coordinates
    points_file = 'site_details.csv'

    # Start/end dates in one of the formats: year (2015) or year and day-of-year (2015032) or year-month-day (20150201)
    start_date = 2015001
    end_date = 2015002

    # Optional arguments
    optional_args = '--id-ref "Custom user" --node-col node_id'

    # Prepare the work item (NOT to be modified).
    docker_image = "weather-files:1.0"
    command_pattern = "python /app/generate_weather_asset_collection.py {} {} {} {}"
    command = command_pattern.format(points_file, start_date, end_date, optional_args)
    user_files = FileList(root='.', files_in_root=[points_file])

    wi = WorkItemManager(item_name=wi_name, docker_image=docker_image, command=command, user_files=user_files,
        tags={'Demo': 'dtk-tools Docker WorkItem', 'WorkItem type': 'Docker', 'Command': command })

    wi.execute()


if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()
    main()

