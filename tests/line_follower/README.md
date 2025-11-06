Write an script to train a line follower. It will have 8 sensors aligned in a single line. I need you to use pygame. I need you to create multiple tracks(many tracks) so that it learns to drive by itself. Then use something like deepq. Please make sure to use some framework that I could compile to then insert into an esp32 or something like that. If possible use pytorch. The car will have two wheels, consider acute corners or more smooth tracks; i need to try many kinds of tracks. Write the code in a single file. Automatically generate the tracks. Use something like deep q learning. 



## Commands
```
python main.py --render 1 --train_episodes 100
python main.py --train_episodes 300 --render 0
python main.py --train_episodes 0 --demo 1 --render 1 --model_path dqn_line_follower.pth
python main.py --train_episodes 0 --demo 1 --render 1 --model_path dqn_line_follower.pth --demo_level all --episodes_per_level 2
python main.py --train_episodes 0 --demo 1 --render 1 --model_path dqn_line_follower.pth --demo_level all --episodes_per_level 2
```


## TODO
- add to the simulation real sizes, so that the model learns with a more real example.
- Add functionality so that first it learns the track and then just do it faster from the beginning
