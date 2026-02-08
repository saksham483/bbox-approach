simple_pid uses simple pid as name suggest 


cascaded use cascaded pid where pid controls velocity instead of pwm but has a function that can map velocity to pwm 


hybrid uses the hybrid logic:  
Yaw: Cascaded (Vision Error Target Rate Pixhawk).  
Surge: Simple (Vision Error PWM).  
Heave: Simple/Assisted (Vision Error Throttle Offset).
