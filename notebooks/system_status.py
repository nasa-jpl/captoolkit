def tell_system_status():
        # from https://www.programcreek.com/python/example/53878/psutil.disk_usage
        import psutil
        import platform
        import datetime

        os, name, version, _, _, _ = platform.uname()
        version = version.split('-')[0]
        cores = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory().total >> 30
        memory_percent = psutil.virtual_memory()[2]
        disk_percent = psutil.disk_usage('/')[3]
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        running_since = boot_time.strftime("%A %d. %B %Y")
        response = "I am currently running on %s version %s.  \n" % (os, version)
        response += "This system is named %s \n" % (name)
        response += "It has %s CPU cores.  \n" % (cores)
        response += "It has %s Gigabytes of RAM.  \n" % (ram)
        response += "Current disk_percent is %s percent.  \n" % disk_percent
        response += "Current CPU utilization is %s percent.  \n" % cpu_percent
        response += "Current memory utilization is %s percent. \n" % memory_percent
        response += "it's running since %s. \n" % running_since
        return response 
    
print(tell_system_status())
