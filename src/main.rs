//following rust tutorial for now before NN
use std::fs

fn optionenum(){
    //if we have functions that return should return a Null, instead we return an optional type, which will be the proper type, or a none type, but option wraps the null type so its safer.
    // to return an optional value, we return  Some(value) or none, since some makes an option of value, not just the value, which makes the return value an enum
    // 


}

fn errorhandling(){
    //Error Handling in rust. Uses two enums
    //Use the result Enum, we can have a good result type, or an error result type, but the error could be any number of things and then good result could be any number of data types, so instead we use a genertic.
    //Generic is like type T, T can be anything. Captial Letters are generic fill in types 
    //Rust provides the result enum for us already, with the Okay type, and the Error type


    //~~ 
    enum Result<A,B>{
        Ok(A),
        Err(B),
    }

    let res = fs::read_to_string("hello");

    match res{
        Ok(contents)=> println!("Contents are: {}",contents),
        Err(error)=> println!("Error: {}",error),
    } //-> so we can use pattern matching, similar to optional, to get the response.

    //using res.unwrap(). unwrap function will return the internal value that we have picked of the reuslt, so it could be valid or error etc.


}

enum Shape{
        Circle(f64),
        Square(f64),
        Triangle(f64, f64, f64),
    }

fn enums() {
    //Enums are like custom variables where things have varients.
    // For example if we want to make a compass variable, we can make a direction enum, and its 4 variants can be N,E,S,W. and then instead of using stringsm we have actual types for that, Its like a multiclass boolean
    enum Direction {
        N,
        E,
        S,
        W,
    }
    //enums make functions etc  more strict and so its better. You have specific varients instead of just using any const strings.


    let dir = Direction::N;
    match dir {
        Direction::N => println!("North"),
        Direction::E => println!("East"),
        Direction::S => println!("South"),
        Direction::W => println!("West"),
    }

    //you can also have enums with varients that has data, like shapes
    enum Shape{
        Circle(f64),
        Square(f64),
        Triangle(f64, f64, f64),
    }

    let circ = Shape::Circle(5.0);

    //best way to do logic with enums is pattern matchin enums

    // this is like a sin

}

//paternmatching
fn calculateArea(shape: Shape)->f64{
    match shape{
        Shape::Circle(rad) => rad*rad*3.14,
        Shape::Square(side) => side*side,
        Shape::Triangle(b,h) => 0.5*b*h,
    }
}

struct User {
    active: bool,
    username: String,
    email: String,
    age: u32,
}

fn implementing() {
    //we can also use impl "Struct Name{...}"}

    //where we can define functions as a part of the struct to act like classes in c++,
    //we dp:
    /*
    struct Rect{
        width: u32,
        height: u32,
    }

    impl Rect{
        fn get_area(&self) -> u32{
            self.width * self.height //if the last line , XXWERWERWEX, doesnt have a semicolor it is a shortcut for return XXWERWERWEX;
        }
        fn get_peri(&self) -> u32{
            return 2* self.width + 2* self.height;
        }
    }

    we also have debug and traits, for examlple to display we need format (fmt)

    impl Debug for Rect{
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f,"{}",self.width*self.height) //this will return a write message or object of the area when we want to display the rect it self
        }


    We can also make unit struct which are variable/attributeless structs, they only have impls so we have functions to run without private vars

     */
}

fn structures() {
    let user = User {
        username: String::from("Parth"),
        age: 30,
        active: true,
        email: String::from("parth@gmail.com"),
    };

    //if the key and value of hte variable and the struct is the same, then instead of assigning it, we can just use ir ie:
    let username = String::from("me");

    let user2 = User {
        username,
        active: true,
        email: String::from("dud@gmail.com"),
        age: 34,
    };
}

fn borrowing() {
    // //borrowing
    // // variables can be borrowed by other guys but there will still be only one single owner.
    // // can be borrowed by many, but if we borrow and modify, then only one borrower has access, and the others cant borrow it.
    // //if a function called is not mutatating the guy, then the function just borrows the data, , its immutable borrowing, and goes bcak to owner after borrower is dead.
    // //pass by reference wil pass &addr and so we dont give the ownership, just address
    // let mut s1:String = String::from("hello dudew");
    // let s2 = &s1;

    // println!("1{}",s1);
    // println!("2{}",s2);

    // take_ownership(s2);
    // println!("3{}",s1);

    // //mutable references
    // let mut s1:String = String::from("Hello");
    // update_string(&mut s1);
    // println!("4{}",s1);

    //cant hvae multiple mutable refs or any immuts after mut ref

    let mut s3 = String::from("Hekki");

    let esss = &mut s3;
    println!("{}", esss);
    let bes = &mut s3;

    println!("5{}", bes);
}

fn update_string(string: &mut String) {
    string.push_str("dude");
}

fn take_ownership(s: &String) {
    println!("2.5{}", s);
}

fn ownership() {
    //mem management w heap in rust
    //everythin is immutable by default, it stops race conditions between threads, unless stateed to be mutable (mut)

    //ownership in Rust
    /*
    ownership is a set of rules for rust to managem memor. -> Set of rules for the compiler to check to make sure that no errors will be possible at compile time in the code.
    stops dangling pts errors etc like in c++.

    ownership makes program faster, but makes compilation slower. compliation is slower for rust but runtime vry fast,
    ownerhsip is there for things on the heap mostly not stack

    data must always have one boyfriend. boyfriends can be changed and passed on. if no bf, then i will die.

    stack variables on the stack dont really matter.
    they are owned by the calling function that they are in. that function that they are made in is the owner.
    the inside variables go out of scope when the calling function stack frame is popped off the stack


    heap:
    anytime something is created on the heap, we need a pointer on the stack to it. That is the owner, (how does work. what defines a variable, like a matrix has one owner or nxn owner,s what if i put a object inside of another)
    also rust stopts double owners, if i have int* s1 = new 4, s2 = s1. both s1 and s2 would point to 4, but now the owner of 4 changes, the variable moves on and the old owner becomes invalid, so the object/variable can only retain
    a single owner at all times. if we have multiple, the olderones gets ignored and (killed i think so) and only the last one is considered the owner, so when it goes away, 4 will too.
    you can not have two boyfriends.  avoiding double owners in heap = no double free error

     */
    let s1: String = String::from("hello");
    println!("{}", s1);
    //this is fine because we havent moved
    let s2: String = s1;
    //println!("{}",s1); //error since s1 is no longer the owner and so we cant access it. The value is being borrowed after it was already moved
    println!("{}", s2);
    //rust says that whenever the current owner of a data object dies, then the heap automatically dies, so we dont need manual Mem Manage or garbage collector, instead the owners pet automatically dies anytime an owner dies

    //values can also be moved or borrowed by going into a function (cheating on their owner) when you pass a variable it gets copied fully cuz its only on the stack, but for heap variables, we create a new stack pointer to it, but its the same heaping girlfriend variable

    //if you pass in a string, you copy a pointer to its loc on the heap as the new guy, but its still the same char[].

    //if you use object.clone(), then we allocated a new girlfriend instead of the same on, so we are chill, the twin has a new bf, but the og bf stays.
    //you canreturn owner ship via returning something back to the caller function and then it my_string will take back ownership from stealerstring which was passed into a function, returning doesnt bounce back but instead just tells the compiler that this is the new owner.
}

fn loops() {
    //loops and conditionals
    let is_even = true;

    if is_even {
        println!("This is even!");
    } else if !is_even {
        println!("This is not even");
    } else {
        println!()
    }

    // for i in 0..100{
    //     print!("{}",i);
    // }

    let sentence: String = String::from("my name is parth");
    let firstword = get_first_word(sentence);
    println!("The first word is: {}", firstword);

    let n = 50;
    for _i in 0..n {
        println!("Hello dude");
    }
}

fn get_first_word(sentence: String) -> String {
    //you must define the return type of the function whenever you create it.
    let mut ans: String = String::from("");
    for char in sentence.chars() {
        if char == ' ' {
            break;
        }
        ans.push_str(char.to_string().as_str());
    }
    return ans;
}

fn strings() {
    let x: &str = "strisngs";

    //not a fixed type cuz there are not fixed length like vector so its hard to keep it
    //since it can change at runtime,

    //string mutation is hard for rust because you have to change the allocated space, not automatica, unlike nums which have fixed space
    //instead use string::from("...")
    let mut words: String = String::from("hello ");

    let greet = String::from("hello world");

    println!("{}", greet);

    let char1: Option<char> = greet.chars().nth(0); //so the type is optionally a character, or it will be none, so rust wont let us directly print this char, we have to use match/switch for the case if it is and hte case if its not.

    match char1 {
        //pattern matching
        Some(c) => print!("{}", c),
        None => print!("No char at this location"),
    }

    print!("char1:{}", char1.unwrap()) //unwrap will unwrap the option type and so we will be chill. it will cause a runtime exception
}

fn bool() {
    let mut is_male = false;

    //let isnotmale = !is_male;

    if is_male {
        print!("You are male");
    } else {
        is_male = true;
    }
    if is_male {
        print!("Now you are male")
    }
}

fn vars() {
    //simple variables in rust
    //define vars with let
    let x = 5;
    //let is the init-er
    let w: i32 = 6; //we can also declare the type of the variable we want to make it. //these are immutable by default

    //    let z:u8 = -4; //error because unsigned cant be negative
    let y: u32 = 1000;

    let f: f32 = 1.001; // float

    print!("x :{}", f);

    //runtime overflow: cant be checked by compiler obviously
}
