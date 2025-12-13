# -*- encoding:utf-8 -*-

echo "-------------------skilled forgery: stylus evaluation begin-------------------"
python evaluate_stylus.py --epoch 39 --template_num 4 --weights folder
python verify_stylus.py --template_num 4 --epoch 39 --weights folder
python verify_stylus.py --template_num 1 --epoch 39 --weights folder
echo "-------------------skilled forgery: stylus evaluation done-------------------"

echo "-------------------skilled forgery: finger evaluation begin-------------------"
python evaluate_finger.py --epoch 39 --template_num 4 --weights folder
python verify_finger.py --template_num 4 --epoch 39 --weights folder
python verify_finger.py --template_num 1 --epoch 39 --weights folder
echo "------------------skilled forgery: finger evaluation done-------------------"

echo "-------------------random forgery: stylus evaluation begin-------------------"
python evaluate_stylus.py --epoch 39 --template_num 4 --weights folder --rf
python verify_stylus.py --template_num 4 --epoch 39 --weights folder
python verify_stylus.py --template_num 1 --epoch 39 --weights folder
echo "-------------------random forgery: stylus evaluation done-------------------"

echo "-------------------random forgery: finger evaluation begin-------------------"
python evaluate_finger.py --epoch 39 --template_num 4 --weights folder --rf
python verify_finger.py --template_num 4 --epoch 39 --weights folder
python verify_finger.py --template_num 1 --epoch 39 --weights folder
echo "------------------random forgery: finger evaluation done-------------------"